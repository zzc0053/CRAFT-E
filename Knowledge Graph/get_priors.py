from collections import defaultdict
import numpy as np
def parse_graph(file_path):
	"""Parse the verb-property-object graph text file into dictionaries."""
	verb_to_properties = defaultdict(list)
	property_to_objects = defaultdict(list)

	with open(file_path, 'r') as f:
		for line in f:
			parts = line.strip().split(',')
			if len(parts) != 3:
				continue
			source, target, score = parts
			score = float(score)

			# Classify line: verb->property OR property->object
			if source.isalpha() and target.isalpha():  # basic heuristic, verbs usually first
				if source in verb_list:  # if verb, property line
					verb_to_properties[source].append((target, score))
				else:
					property_to_objects[source].append((target, score))
					# property_to_objects[target].append((source, score))

	return verb_to_properties, property_to_objects


def get_top_objects_for_verbs(verb_to_properties, property_to_objects, top_k=5):
	"""Aggregate objects for each verb using weighted property-object links."""
	verb_to_objects = {}

	for verb, props in verb_to_properties.items():
		object_scores = defaultdict(list)
		for prop, v_score in props:
			if prop in property_to_objects:
				for obj, p_score in property_to_objects[prop]:
					object_scores[obj] += [v_score * p_score]
				for obj, p_score in property_to_objects[prop]:
					object_scores[obj] = np.linalg.norm(object_scores[obj])

		# Sort objects by score and take top_k
		sorted_objs = sorted(object_scores.items(), key=lambda x: x[1], reverse=True)
		verb_to_objects[verb] = [obj for obj, score in sorted_objs[:top_k]]
		# print([(obj,score) for obj, score in sorted_objs[:top_k]])

	return verb_to_objects


# Example usage:
# verb_list = ["wave", "lift", "play", "break", "open", "pull", "hang", "insert", "don", "push", "rotate", "construct", "cut", "contain", "surround", "cover", "build", "prepare", "pick", "eat", "enclose", "produce", "protect", "serve", "retrieve", "transfer", "wear", "carry", "write", "hit", "hear", "paint", "transport", "wrap", "feed", "activate", "install"]  # provide list of verbs

# verb_list = ['activate', 'break', 'build', 'carry', 'construct', 'contain', 'cover', 'cut', 'defend', 'deliver', 'design', 'don', 'eat', 'enclose', 'feed', 'grow', 'hang', 'hear', 'hit', 'insert', 'install', 'lift', 'open', 'paint', 'perform', 'pick', 'play', 'prepare', 'produce', 'protect', 'pull', 'push', 'put', 'read', 'retrieve', 'rotate', 'serve', 'support', 'surround', 'test', 'tour', 'transfer', 'transport', 'tune', 'visit', 'wave', 'wear', 'win', 'wrap', 'write']

verb_list = ["play", "contain", "construct", "support", "design", "serve", "transfer", "grow", "install", "activate", "cover", "cut", "write", "feed", "pick", "open", "wear", "pull", "eat", "rotate"]
file_path = "verb_property_object_graph_realworld_v3.txt"
verb_to_properties, property_to_objects = parse_graph(file_path)
top_objects = get_top_objects_for_verbs(verb_to_properties, property_to_objects, top_k=50)

import json
with open("top_objects_per_verb_realworld.json", "w") as f:
	json.dump(top_objects, f, indent=2)

# print(json.dumps(top_objects, indent=2))