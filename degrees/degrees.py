import csv
import sys
from collections import deque

# Maps names to a set of corresponding scientist_ids
names = {}

# Maps scientist_ids to a dict of: name, set of paper_ids
people = {}

# Maps paper_ids to a dict of: title, year, set of author_ids
papers = {}


class Node:
    def __init__(self, state, parent, action):
        self.state = state  # scientist_id
        self.parent = parent  # Node
        self.action = action  # paper_id


def load_data(directory):
    """
    Load data from CSV files into memory.
    """
    # Load scientists
    with open(f"{directory}/scientists.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            people[row["scientist_id"]] = {
                "name": row["name"],
                "papers": set()
            }
            name_key = row["name"].lower()
            names.setdefault(name_key, set()).add(row["scientist_id"])

    # Load papers
    with open(f"{directory}/papers.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers[row["paper_id"]] = {
                "title": row["title"],
                "year": row["year"],
                "authors": set()
            }

    # Load authorship
    with open(f"{directory}/authors.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                people[row["scientist_id"]]["papers"].add(row["paper_id"])
                papers[row["paper_id"]]["authors"].add(row["scientist_id"])
            except KeyError:
                continue


def shortest_path(source, target):
    """
    Find shortest path between two scientists using BFS.
    Returns list of (paper_id, scientist_id) tuples.
    """
    start = Node(source, None, None)
    frontier = deque([start])
    explored = set()
    frontier_ids = {source}

    while frontier:
        node = frontier.popleft()
        frontier_ids.remove(node.state)
        explored.add(node.state)

        for paper_id, neighbor_id in neighbors_for_person(node.state):
            if neighbor_id in explored or neighbor_id in frontier_ids:
                continue

            child = Node(neighbor_id, node, paper_id)

            if neighbor_id == target:
                path = []
                while child.parent is not None:
                    path.append((child.action, child.state))
                    child = child.parent
                path.reverse()
                return path

            frontier.append(child)
            frontier_ids.add(neighbor_id)

    return None


def neighbors_for_person(person_id):
    """
    Returns (paper_id, person_id) pairs for co-authors of a given scientist.
    """
    neighbors = set()
    for paper_id in people[person_id]["papers"]:
        for author_id in papers[paper_id]["authors"]:
            if author_id != person_id:
                neighbors.add((paper_id, author_id))
    return neighbors


def person_id_for_name(name):
    """
    Resolves a person's name to a scientist_id.
    """
    person_ids = list(names.get(name.lower(), set()))
    if not person_ids:
        return None
    elif len(person_ids) > 1:
        print(f"Multiple scientists found for '{name}':")
        for pid in person_ids:
            print(f"{pid}: {people[pid]['name']}")
        try:
            selected = input("Intended Person ID: ").strip()
            if selected in person_ids:
                return selected
        except Exception:
            pass
        return None
    else:
        return person_ids[0]


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python degrees.py [directory]")
    directory = sys.argv[1]

    print("Loading data...")
    load_data(directory)
    print("Data loaded.")

    source_name = input("Name: ").strip()
    source = person_id_for_name(source_name)
    if source is None:
        sys.exit(f"Scientist '{source_name}' not found.")

    target_name = input("Name: ").strip()
    target = person_id_for_name(target_name)
    if target is None:
        sys.exit(f"Scientist '{target_name}' not found.")

    path = shortest_path(source, target)

    if path is None:
        print("No connection found.")
    else:
        print(f"{len(path)} degrees of separation.")
        current = source
        for i, (paper_id, person_id) in enumerate(path, 1):
            scientist1 = people[current]["name"]
            scientist2 = people[person_id]["name"]
            paper = papers[paper_id]["title"]
            print(f"{i}: {scientist1} and {scientist2} co-authored \"{paper}\"")
            current = person_id


if __name__ == "__main__":
    main()
