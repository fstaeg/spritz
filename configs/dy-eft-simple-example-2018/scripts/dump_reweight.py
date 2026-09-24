import argparse
import json

def parse_reweight_card(path):
    results = {}
    current_name = None
    count = 0

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("launch --rwgt_name="):
                current_name = line.split("launch --rwgt_name=")[1].strip()
                results[current_name] = {"idx": count, "values": []}
                count += 1
                continue

            if line.startswith("set") and current_name:
                tokens = line.split()
                param, value = " ".join(tokens[1:-1]), tokens[-1]
                if float(value) != 0:
                    results[current_name]["values"].append([param, value])

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse MadGraph reweight card")
    parser.add_argument("input", help="Path to input .dat file")
    parser.add_argument("-o", "--output", help="Output JSON file", default="rwgt.json")
    args = parser.parse_args()

    parsed = parse_reweight_card(args.input)

    with open(args.output, "w") as f:
        json.dump(parsed, f, indent=2)

    print(f"Saved {len(parsed)} reweight points to {args.output}")

