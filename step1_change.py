import json
import os
import shutil
from sklearn.metrics import accuracy_score, f1_score

def process_data(json_path, output_json_path, target_root):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    true_labels = []
    pred_labels = []

    # 用于存储更新之后的所有数据
    new_data = []

    for idx, item in enumerate(data):
        ground_truth = item.get("ground_truth", "")
        model_output = item.get("model_output", "")
        original_image_path = item.get("original_image_path", "")
        modified_image_path = item.get("modified_image_path", None)
        
        # 1) 计算二分类标签
        # ground_truth_label: 0 表示 "nothing"，1 表示非 "nothing"
        # import pdb
        # pdb.set_trace()
        if ground_truth == "nothing has been modified in this image.":
            true_label = 0
        else:
            true_label = 1
        
        # pred_label: 同理
        if model_output == "nothing has been modified in this image." or "nothing" in model_output:
            pred_label = 0
        else:
            pred_label = 1
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)

        # 2) 搬运图片 & 更新路径
        if modified_image_path is not None:
            # 这里可以根据 original_image_path 和 modified_image_path 来构造一个唯一文件夹名
            # 比如直接用它们文件名拼接，或者再加一个 idx
            orig_name = os.path.basename(original_image_path)
            mod_name = os.path.basename(modified_image_path)
            folder_name = f"{os.path.splitext(orig_name)[0]}_{os.path.splitext(mod_name)[0]}_{idx}"
            new_folder_path = os.path.join(target_root, folder_name)
            
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            
            # 拷贝 original_image_path 到 new_folder_path
            new_orig_path = os.path.join(new_folder_path, orig_name)
            shutil.copy2(original_image_path, new_orig_path)
            
            # 拷贝 modified_image_path 到 new_folder_path
            new_mod_path = os.path.join(new_folder_path, mod_name)
            shutil.copy2(modified_image_path, new_mod_path)
            
            # 更新 item 中的路径
            item["original_image_path"] = new_orig_path
            item["modified_image_path"] = new_mod_path
        
        new_data.append(item)

    # 计算 accuracy, f1-score
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    # 打印指标
    print("Accuracy:", acc)
    print("F1-score:", f1)

    # 写回新的 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 假设原始 JSON 路径
    json_file = "/fdata/FragFake/finished_file/dataset/step1xedit/result/step1xedit_hard_gemma3_4b.json"
    # 输出更新后 JSON 的路径
    output_json_file = "zzy_anno_step1xedit_hard_gemma3_4b.json"
    # 搬运图片的目标根目录
    target_root_folder = "/data_sda/zzy/pic_anno"

    process_data(json_file, output_json_file, target_root_folder)
