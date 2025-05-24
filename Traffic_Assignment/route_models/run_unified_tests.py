import os
import subprocess
import re
import csv

test_dir = "unified_tests"
tests = sorted([f for f in os.listdir(test_dir) if f.endswith(".txt")])
algorithms = [
    "bfs_search.py",
    "dfs_search.py",
    "gbfs_search.py",
    "astar_search.py",
    "cus1_search.py",
    "cus2_search.py"
]

results = []

def parse_testfile(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    nodes, edges, origin, destinations = {}, {}, None, []
    section = None
    for line in lines:
        if line.startswith("Nodes:"): section = "nodes"
        elif line.startswith("Edges:"): section = "edges"
        elif line.startswith("Origin:"): section = "origin"
        elif line.startswith("Destinations:"): section = "dest"
        elif section == "nodes":
            k, v = line.split(":")
            nodes[int(k.strip())] = eval(v.strip())
        elif section == "edges":
            k, v = line.split(":")
            edges[eval(k.strip())] = int(v.strip())
        elif section == "origin":
            origin = int(line)
        elif section == "dest":
            destinations = [int(d.strip()) for d in line.split(";")]
    return nodes, edges, origin, destinations

def inject_and_run(algo_file, test_file, nodes, edges, origin, destinations):
    with open(algo_file, "r", encoding="utf-8") as f:
        code = f.read()

    def dict_to_str(d):
        return "{\n" + ",\n".join(f"    {repr(k)}: {repr(v)}" for k, v in d.items()) + "\n}"

    code = re.sub(r"nodes\s*=\s*{.*?}", f"nodes = {dict_to_str(nodes)}", code, flags=re.DOTALL)
    code = re.sub(r"edges\s*=\s*{.*?}", f"edges = {dict_to_str(edges)}", code, flags=re.DOTALL)
    code = re.sub(r"origin\s*=\s*\d+", f"origin = {origin}", code)
    code = re.sub(r"destinations\s*=\s*\[.*?\]", f"destinations = {destinations}", code)

    temp_file = f"temp_{algo_file}"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(code)

    completed = subprocess.run(["python", temp_file], capture_output=True, text=True)
    output = completed.stdout.strip().split("\n")
    os.remove(temp_file)


    i = 0
    while i < len(output):
        if i + 2 >= len(output):
            break  # 不完整结果块
        try:
            fname, method = output[i].strip().split()
            goal, nodes_created = output[i+1].strip().split()
            path_line = output[i+2].strip()
            cost_line = output[i+3].strip() if (i+3 < len(output)) and "Cost:" in output[i+3] else "N/A"

            results.append({
                "Test": test_file,
                "Algorithm": method.upper(),
                "Goal": goal,
                "Nodes_Created": int(nodes_created),
                "Path": path_line,
                "Cost": cost_line.replace("Cost:", "").strip() if "Cost:" in cost_line else "N/A"
            })
            i += 4 if "Cost:" in cost_line else 3
        except Exception as e:
            i += 1
            continue

for algo in algorithms:
    for test_file in tests:
        path = os.path.join(test_dir, test_file)
        nodes, edges, origin, destinations = parse_testfile(path)
        inject_and_run(algo, test_file, nodes, edges, origin, destinations)

with open("unified_test_results.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Test", "Algorithm", "Goal", "Nodes_Created", "Path", "Cost"])
    writer.writeheader()
    writer.writerows(results)

print("✅ Results saved to unified_test_results.csv")