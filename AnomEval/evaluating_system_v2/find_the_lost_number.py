import json

file_path_0 = r'/mnt/new_disk/tpami/models/ours/ours_lunwen_outcome.json'
file_path_1 = r'/mnt/new_disk/tpami/models/chat-univi/result/chatunivi_new_outcome.json'

with open(file_path_0, 'r', encoding='utf-8') as f:
    data = json.load(f)


existing_files = []
for entry in data:
    file_name = entry['video_file']
    file_number = int(file_name.split('.')[0])
    existing_files.append(file_number)


all_files = set(range(1, 1001))
missing_files = all_files - set(existing_files)

print(f"Missing files: {sorted(missing_files)}")


for i in missing_files:
    data.append({
        "video_file": f"{str(i).zfill(5)}.mp4",
        "prompt": "This is a placeholder description for video number {i}.",
        "output": "This is a placeholder output for video number {i}.",
        "task_type": "default"})

with open(file_path_0, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
