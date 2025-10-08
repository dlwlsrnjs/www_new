import math
import random
import re
from collections import defaultdict

from submission.vendor.rekgmcts.prompts import EVALUATE_STATE_PROMPT, entity_p_prompt
from submission.vendor.rekgmcts.utils import extract_entity_names


class MCTSNode:
    def __init__(self, entities_info, triples=None, pre_relations=None, pre_head=None, parent=None):
        self.entities_info = entities_info
        self.parent = parent
        self.children = []
        self.y = triples or []
        self.v = 0.0
        self.pre_relations = pre_relations or []
        self.pre_head = pre_head if pre_head is not None else -1
        self.visits = 1
        self.total_reward = 0
        self.is_fully_expanded = False
        self.unexpanded_entities = list(range(len(entities_info)))
        self.cached_relations = {}
        self.expanded_relations = {entity['entity_id']: set() for entity in entities_info}

    def add_child(self, entity_idx, new_entity_info, new_triple, relation, head):
        new_entities = self.entities_info.copy()
        new_entities.pop(entity_idx)
        new_entities.append(new_entity_info)
        new_triples = self.y + [new_triple]
        child = MCTSNode(
            entities_info=new_entities,
            triples=new_triples,
            pre_relations=[relation],
            pre_head=head,
            parent=self,
        )
        child.cached_relations = self.cached_relations.copy()
        child.expanded_relations = {entity['entity_id']: set() for entity in new_entities}
        self.children.append(child)
        self.expanded_relations[self.entities_info[entity_idx]['entity_id']].add((relation, head))
        return child

    def cache_relations(self, entity_id, relations):
        self.cached_relations[entity_id] = relations

    def get_cached_relations(self, entity_id):
        return self.cached_relations.get(entity_id, [])

    def mark_entity_fully_expanded(self, entity_idx):
        if entity_idx in self.unexpanded_entities:
            self.unexpanded_entities.remove(entity_idx)
            if not self.unexpanded_entities:
                self.is_fully_expanded = True

    def get_unexpanded_entity(self):
        if self.unexpanded_entities:
            idx = self.unexpanded_entities[0]
            return idx, self.entities_info[idx]
        return None, None

    def get_uct_value(self, exploration_constant):
        if self.visits == 0:
            return float('inf')
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


def _construct_triple(source_entity, relation, target_entity, is_head):
    if is_head:
        return (source_entity, relation, target_entity)
    else:
        return (target_entity, relation, source_entity)


class MCTSPathFinder:
    def __init__(self, question, topic_entities, llm, num_retain_entity=5, num_retain_relation=5, max_depth=5, max_iterations=5, score_threshold=0.8, exploration_constant=0.5, prune_min_candidates=2):
        self.question = question
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        self.alpha = 0.5
        self.llm = llm
        self.num_retain_entity = num_retain_entity
        self.num_retain_relation = num_retain_relation
        self.score_method = 'none'
        self.exploration_constant = exploration_constant
        self.prune_min_candidates = prune_min_candidates
        entities_info = [{'entity_id': id, 'entity_name': name} for id, name in topic_entities.items()]
        self.root = MCTSNode(entities_info=entities_info, pre_relations=[], pre_head=-1)
        self._eval_cache = {}
        self._prune_cache = {}

    def search(self):
        iterations = 0
        best_node = None
        best_value = float('-inf')
        while iterations < self.max_iterations:
            node = self._select(self.root)
            if not node:
                break
            children = self._expand(node)
            if children:
                if len(children) == 1 and children[0].v >= 0.9:
                    return self._extract_path(children[0])
                if self.score_method == 'vm':
                    best_child = max(children, key=lambda x: x.v)
                    simulated_v = self._simulate(best_child)
                    best_child.v = best_child.v * (1 - self.alpha) + simulated_v * self.alpha
                    best_child.visits += 1
                self._backpropagate(node)
                for child in children:
                    if child.v > best_value:
                        best_value = child.v
                        best_node = child
                    if self._check_solution(child):
                        return self._extract_path(child)
            iterations += 1
        best_node = self._get_best_node() if best_node is None else best_node
        return self._extract_path(best_node) if best_node else []

    def evaluate(self, triples):
        question = self.question
        key = tuple(triples)
        if key in self._eval_cache:
            return self._eval_cache[key]
        formatted_triples = []
        for head_name, relation, tail_name in triples:
            formatted_triple = f"{head_name}, {relation}, {tail_name}"
            formatted_triples.append(formatted_triple)
        triples_str = "\n".join(formatted_triples)
        formatted_prompt = EVALUATE_STATE_PROMPT.format(question=question, triple=triples_str)
        formatted_prompt += "\nSTRICT OUTPUT: On the first line, output only the numeric rating between 0.0 and 1.0 with one decimal (e.g., 0.8). Then give one short justification line."
        value_eval = self.llm(formatted_prompt)[0]
        score_match = re.search(r"(\d+(?:\.\d+)?)", value_eval)
        if score_match:
            try:
                score = float(score_match.group(1))
            except Exception:
                score = 0.0
        else:
            score = 0.0
        score = max(0.0, min(1.0, score))
        self._eval_cache[key] = score
        return score

    def entity_prune(self, candidate_entities, node, relation):
        candidate_names = [get_entity_name(eid) for eid in candidate_entities]
        matched_entity_ids = []
        name_to_ids = defaultdict(list)
        for name, eid in zip(candidate_names, candidate_entities):
            name_to_ids[name].append(eid)
        current_entities = ', '.join([current_entity['entity_name'] for current_entity in node.entities_info])
        path_history = ' -> '.join([f"{t[0]}-{t[1]}-{t[2]}" for t in node.y])
        prompt_str = entity_p_prompt.format(
            top_k=self.num_retain_entity,
            question=self.question,
            current_entities=current_entities,
            current_relation=relation,
            path_history=path_history,
            candidate_names=', '.join(candidate_names)
        )
        cache_key = (relation, current_entities, path_history, tuple(candidate_names))
        if cache_key in self._prune_cache:
            llm_output = self._prune_cache[cache_key]
        else:
            llm_output = llm(prompt_str)[0]
            self._prune_cache[cache_key] = llm_output
        entities = extract_entity_names(llm_output)
        for name in entities:
            if name in name_to_ids and name_to_ids[name]:
                selected_id = random.choice(name_to_ids[name])
                matched_entity_ids.append(selected_id)
                name_to_ids[name].remove(selected_id)
                if len(matched_entity_ids) >= self.num_retain_entity:
                    break
        if len(matched_entity_ids) < self.num_retain_entity:
            for eid in candidate_entities:
                if eid not in matched_entity_ids:
                    matched_entity_ids.append(eid)
                    if len(matched_entity_ids) >= self.num_retain_entity:
                        break
        return matched_entity_ids

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded and node.children:
            node = self._get_best_child(node)
        if self._is_terminal(node):
            return None
        return node

    def _expand(self, node: MCTSNode):
        children = []
        for entity_idx, entity_info in enumerate(node.entities_info):
            entity_id = entity_info['entity_id']
            if entity_id in node.cached_relations:
                pass
            else:
                relations = relation_search_prune(
                    entity_id,
                    entity_info['entity_name'],
                    node.pre_relations,
                    node.pre_head,
                    self.question,
                    self.llm
                )
                node.cache_relations(entity_id, relations)
            relations = node.get_cached_relations(entity_id)
            for relation_info in relations:
                if (relation_info['relation'], relation_info['head']) in node.expanded_relations[entity_id]:
                    continue
                target_entities = entity_search(
                    entity_id,
                    relation_info['relation'],
                    relation_info['head']
                )
                if len(target_entities) >= self.prune_min_candidates:
                    target_entities = self.entity_prune(target_entities, node, relation_info['relation'])
                for target_id in target_entities:
                    target_name = get_entity_name(target_id)
                    if any(target_id == e['entity_id'] for e in node.entities_info):
                        continue
                    new_triple = _construct_triple(entity_info['entity_name'], relation_info['relation'], target_name, relation_info['head'])
                    child = node.add_child(
                        entity_idx=entity_idx,
                        new_entity_info={'entity_id': target_id, 'entity_name': target_name},
                        new_triple=new_triple,
                        relation=relation_info['relation'],
                        head=relation_info['head']
                    )
                    child.v = self.evaluate(child.y)
                    children.append(child)
                    if child.v >= 0.9:
                        return [child]
                node.expanded_relations[entity_id].add((relation_info['relation'], relation_info['head']))
            if len(node.expanded_relations[entity_id]) == len(relations):
                node.mark_entity_fully_expanded(entity_idx)
        if all(len(expanded) == len(node.get_cached_relations(entity['entity_id'])) for entity, expanded in zip(node.entities_info, node.expanded_relations.values())):
            node.is_fully_expanded = True
        return children

    def _simulate(self, node: MCTSNode, roll_forward_steps: int = 3) -> float:
        max_value = node.v
        current_path = node.y.copy()
        entity_idx = random.randrange(len(node.entities_info))
        current_entity = node.entities_info[entity_idx]
        pre_relations = node.pre_relations.copy()
        pre_head = node.pre_head
        for _ in range(roll_forward_steps):
            relations = relation_search_prune(
                current_entity['entity_id'],
                current_entity['entity_name'],
                pre_relations,
                pre_head,
                self.question,
                self.llm
            )
            if not relations:
                break
            relation_info = relations[0]
            target_entities = entity_search(
                current_entity['entity_id'],
                relation_info['relation'],
                relation_info['head']
            )
            if not target_entities:
                break
            target_id = target_entities[0]
            target_name = get_entity_name(target_id)
            new_triple = {
                'subject': current_entity['entity_name'],
                'relation': relation_info['relation'],
                'object': target_name,
                'head': relation_info['head']
            }
            current_path.append(new_triple)
            try:
                current_value = self.evaluate(self.question, current_path)
                max_value = max(max_value, current_value)
            except Exception:
                break
            if current_value >= self.score_threshold:
                break
            current_entity = {'entity_id': target_id, 'entity_name': target_name}
            pre_relations.append(relation_info['relation'])
            pre_head = relation_info['head']
        return max_value

    def _backpropagate(self, node: MCTSNode):
        current = node
        while current is not None:
            current.visits += 1
            current = current.parent

    def _get_best_child(self, node: MCTSNode) -> MCTSNode:
        return max(node.children, key=lambda x: x.get_uct_value(self.exploration_constant))

    def _is_terminal(self, node: MCTSNode) -> bool:
        return len(node.y) >= self.max_depth or node.is_fully_expanded and not node.children

    def _check_solution(self, node: MCTSNode) -> bool:
        return node.v >= self.score_threshold and len(node.y) <= self.max_depth

    def _extract_path(self, node: MCTSNode):
        return node.y

    def _get_best_node(self):
        def get_best_v(node):
            if not node.children:
                return node, node.v
            max_v = node.v
            max_node = node
            for child in node.children:
                child_node, child_v = get_best_v(child)
                if child_v > max_v:
                    max_v = child_v
                    max_node = child_node
            return max_node, max_v
        best_node, _ = get_best_v(self.root)
        return best_node


