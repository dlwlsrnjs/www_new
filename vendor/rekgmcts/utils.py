import re


def extract_entity_names(output):
    cleaned_output = output.strip()
    entity_names = [name.strip() for name in cleaned_output.split(',')]
    entity_names = [name for name in entity_names if name]
    return entity_names


