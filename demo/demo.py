import base64
from io import BytesIO
from PIL import Image
import torch
import os
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
import re
import numpy as np
import requests
import random
from vis import Visualizer

# utils
def cv2_to_base64(image, ext='.jpg'):
    success, buffer = cv2.imencode(ext, image)
    if not success:
        return None
    return base64.b64encode(buffer).decode('utf-8')

def match_openseg(text):
    pattern = r"<ref>(?!<OTHERS></ref>)(.*?)</ref>"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches and text.find('<mask>') >= 0:
        return True, matches
    else:
        return False, []

def parse_det(text):
    results = []
    parts = text.split('<ref>')
    for part in parts:
        if not part.strip() or '</ref>' not in part:
            continue
        try:
            class_name, content = part.split('</ref>', 1)
            class_name = class_name.strip()
            boxes = re.findall(r'<box>(.*?)</box>', content)
            
            for box_content in boxes:
                matches = re.findall(r'<[xy]_(\d+)>', box_content)
                if len(matches) == 4:
                    coords = list(map(int, matches))
                    results.append({
                        'class': class_name,
                        'bbox': coords
                    })
        except ValueError:
            continue

    results.sort(
        key=lambda item: (item['bbox'][2] - item['bbox'][0]) * (item['bbox'][3] - item['bbox'][1]), 
        reverse=True
    )
    return results

# set target_size smaller can be faster
def crop_and_resize(img, x1, y1, x2, y2, target_size=1280):
    height, width = img.shape[:2]
    padding_scale = 1.2
    crop_width, crop_height = (x2 - x1) * padding_scale, (y2 - y1) * padding_scale
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    x1_ = max(int(center_x - crop_width / 2), 0)
    y1_ = max(int(center_y - crop_height / 2), 0)
    x2_ = min(int(center_x + crop_width / 2), width)
    y2_ = min(int(center_y + crop_height / 2), height)
    img_crop = img[y1_:y2_, x1_:x2_].copy()

    # Draw indicator box for the model (as per original logic)
    adj_x1, adj_y1 = int(x1 - x1_), int(y1 - y1_)
    adj_x2, adj_y2 = int(x2 - x1_), int(y2 - y1_)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.rectangle(img_crop, (adj_x1, adj_y1), (adj_x2, adj_y2), color, thickness=1)

    c_h, c_w = img_crop.shape[:2]
    resize_scale = target_size / min(c_w, c_h)
    new_w, new_h = int(c_w * resize_scale), int(c_h * resize_scale)
    final_img = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    return final_img, resize_scale, [x1_, y1_, x2_, y2_]

def match_refseg(text):
    def parse_polygon(polygon_str):
        match = re.search(r'<ins>(.*?)</ins>', polygon_str, re.DOTALL)
        if match:
            polygon_part = match.group(1)
            points = re.findall(r'<x_(\d+)><y_(\d+)>', polygon_part)
            if not points: return None, None, None, None, None
            coords = [(int(x), int(y)) for x, y in points]
            x_coords, y_coords = zip(*coords)
            return min(x_coords), min(y_coords), max(x_coords), max(y_coords), polygon_part
        return None, None, None, None, None

    x1, y1, x2, y2, _ = parse_polygon(text)
    if x1 is None:
        return False, None, None, None, None
    return True, x1, y1, x2, y2

def parse_prediction_string(pred_string):
    mask_match = re.search(r'<mask>(.*)</mask>', pred_string)
    if not mask_match:
        return np.array([])
    class_names = re.findall(r'<ref>(.*?)</ref>', pred_string)
    pairs = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', pred_string)
    pred_indices = [int(v) for v, n in pairs for _ in range(int(n))]
    pred = np.array(pred_indices, dtype=np.int32)
    if class_names and class_names[0] == "<FG>":
        pred = np.abs(pred - 1)
    return pred

def _get_image_size_from_bytes(img_bytes: bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        return img.width, img.height

def get_image_bytes_and_b64(img_input):
    if img_input is None:
        raise ValueError("img_input is None")
    if isinstance(img_input, (bytes, bytearray, memoryview)):
        img_bytes = bytes(img_input)
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        img_w, img_h = _get_image_size_from_bytes(img_bytes)
        return img_bytes, img_b64, img_w, img_h
    if not isinstance(img_input, str):
        raise TypeError(f"Unsupported img_input type: {type(img_input)}")
    
    s = img_input.strip()
    if s.startswith("http://") or s.startswith("https://"):
        resp = requests.get(s, timeout=10) # Added timeout
        resp.raise_for_status()
        img_bytes = resp.content
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    elif os.path.isfile(s):
        with open(s, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    else:
        try:
            img_bytes = base64.b64decode(s, validate=True)
            img_b64 = s
        except Exception:
            raise ValueError("img_input is not a valid URL, file path, or base64 string")
    
    img_w, img_h = _get_image_size_from_bytes(img_bytes)
    return img_bytes, img_b64, img_w, img_h

def expand_rle_mask_and_replace(s, img_w, img_h):
    m = re.search(r"<mask>\s*(.*?)\s*</mask>", s, re.DOTALL)
    if not m:
        return s
    rle_str = m.group(1)
    pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", rle_str)
    if not pairs:
        return s 
    values = np.fromiter((int(v) for v, _ in pairs), dtype=np.uint16)
    runs   = np.fromiter((int(r) for _, r in pairs), dtype=np.int64)
    pred = np.repeat(values, runs)
    if "<depth>" in s:
        w, h = img_w // 16, img_h // 16
        if pred.size == h * w:
            pred = pred.reshape(h, w)
            pred = cv2.resize(pred, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            pred = np.rint(pred).astype(np.uint16).reshape(-1)
    
    new_mask_str = ",".join(map(str, pred.tolist()))
    return s[:m.start(1)] + new_mask_str + s[m.end(1):]

def build_messages(img_b64, prompt):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_b64},
            {"type": "text", "text": prompt},
        ],
    }]

def process_input(messages):
    new_messages = []
    lst_img = None
    for msg in messages:
        new_content = []
        new_msg = dict(msg)
        if isinstance(msg.get("content"), list):
            for seg in msg["content"]:
                if isinstance(seg, dict) and seg.get("type") == "image_url":
                    url = seg.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        url = url.split(",", 1)[1]
                    new_content.append({"type": "image", "image": url})
                    lst_img = url
                if isinstance(seg, dict) and seg.get("type") == "image":
                    url = seg.get("image", "")
                    new_content.append({"type": "image", "image": url})
                    lst_img = url
                else:
                    new_content.append(seg)
            new_msg["content"] = new_content
        elif isinstance(msg.get("content"), str):
            url = msg.get("content")
            if url.startswith("data:image"):
                url = url.split(",", 1)[1]
                new_content.append({"type": "image", "image": url})
            else:
                new_content.append({"type": "text", "text": url})
            new_msg["content"] = new_content
        new_messages.append(new_msg)
    return new_messages, lst_img
    
# define the model class and inference functions
class YoutuVL(object):
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda", trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        self.seg_crf = True # False can be faster

    def infer(self, messages, max_tokens, temperature, top_p, repetition_penalty, img_input=None):
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        
        generated_ids = self.model.generate(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_tokens,
            img_input=img_input,
            use_crf=self.seg_crf
        )
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        outputs = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return outputs[0]
    
    def infer_segcore(self, img_cv2, x1, y1, x2, y2, pred_mask, class_id, max_tokens, temperature, top_p, repetition_penalty):
        crop_img, _, crop_coords = crop_and_resize(img_cv2, x1, y1, x2, y2)
        crop_img_base64 = cv2_to_base64(crop_img)
        messages_seg = build_messages(crop_img_base64, "Segment the core target.")
        content_seg = self.infer(messages_seg, max_tokens, temperature, top_p, repetition_penalty, crop_img_base64)
        
        crop_h, crop_w = crop_coords[3] - crop_coords[1], crop_coords[2] - crop_coords[0]
        crop_mask = parse_prediction_string(content_seg)
        if crop_mask.any():
            crop_mask = crop_mask.reshape(crop_img.shape[:2])
            crop_mask = cv2.resize(crop_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            roi = pred_mask[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]]
            roi[crop_mask == 1] = class_id
        return pred_mask
    
    def infer_openseg(self, content, img_b64, img_cv2, max_tokens, temperature, top_p, repetition_penalty):
        flag_seg, keywords_category = match_openseg(content)
        if flag_seg:
            prompt = "Detect every objects that belongs to the following categories: " + ",".join(keywords_category)
            messages_qdet = build_messages(img_b64, prompt)
            content_qdet = self.infer(messages_qdet, max_tokens, temperature, top_p, repetition_penalty, img_b64)
            qdet_results = parse_det(content_qdet)[:30]
            if qdet_results:
                class_map = {item: i for i, item in enumerate(keywords_category, 1)}
                pred_mask = np.zeros(img_cv2.shape[:2], dtype=np.uint8)
                for res in qdet_results:
                    x1, y1, x2, y2 = res['bbox']
                    cid = class_map.get(res['class'], 1)
                    pred_mask = self.infer_segcore(img_cv2, x1, y1, x2, y2, pred_mask, cid, max_tokens, temperature, top_p, repetition_penalty)
                mask_flat = ",".join(map(str, pred_mask.reshape(-1)))
                refs = "".join([f'<ref>{item}</ref>' for item in keywords_category])
                return True, f"<ref><BG></ref>{refs}<mask>{mask_flat}</mask>"
        return False, ""
    
    def infer_refseg(self, content, img_b64, img_cv2, max_tokens, temperature, top_p, repetition_penalty):
        flag_refseg, x1, y1, x2, y2 = match_refseg(content)
        if flag_refseg:
            pred_mask = np.zeros(img_cv2.shape[:2], dtype=np.uint8)
            pred_mask = self.infer_segcore(img_cv2, x1, y1, x2, y2, pred_mask, 1, max_tokens, temperature, top_p, repetition_penalty)
            mask_flat = ",".join(map(str, pred_mask.reshape(-1)))
            return True, f"<ref><BG></ref><ref><FG></ref><mask>{mask_flat}</mask>"
        return False, ""
    
    def infer_composite(self, messages, max_tokens, temperature, top_p, repetition_penalty, seg_mode='default'):
        messages, lst_img = process_input(messages)
        content = self.infer(messages, max_tokens, temperature, top_p, repetition_penalty, lst_img)
        if lst_img is None:
            return content
            
        img_data, img_b64, img_w, img_h = get_image_bytes_and_b64(lst_img)
        img_cv2 = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        if seg_mode == 'openseg':
            flag, res = self.infer_openseg(content, img_b64, img_cv2, max_tokens, temperature, top_p, repetition_penalty)
            if flag: return res
            
        flag, res = self.infer_refseg(content, img_b64, img_cv2, max_tokens, temperature, top_p, repetition_penalty)
        if flag: return res
        
        return expand_rle_mask_and_replace(content, img_w, img_h)
    
    def __call__(self, prompt, image_path, max_tokens=4096, temperature=0.1, top_p=0.001, repetition_penalty=1.05, seg_mode="default"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}
        ]
        return self.infer_composite(messages, max_tokens, temperature, top_p, repetition_penalty, seg_mode)

if __name__ == "__main__":
    # init the model and vis function
    model_path = "tencent/Youtu-VL-4B-Instruct" 
    youtu_vl = YoutuVL(model_path)
    vis = Visualizer()

    def run_and_show(task_name, prompt, img_path, is_vis=True, seg_mode='default'):
        print(f"\n{'='*20} [Task: {task_name}] {'='*20}")
        print(f"Prompt: {prompt}")
        
        response = youtu_vl(prompt, img_path, seg_mode=seg_mode)

        if is_vis:
            img = np.array(Image.open(img_path))
            vis_img = vis.visualize(img, response)
            save_path = os.path.join(img_path.split('.')[0] + '_vis.png')
            Image.fromarray(vis_img).save(save_path)
        
        if len(response) > 1000:
            print(f"Response (Truncated): {response[:1000]}...")
        else:
            print(f"Response: {response}")

    #Examples
    run_and_show("VQA", 
                 "How many dogs in the image?", 
                 "assets/1.png",
                 is_vis=False)
        
    run_and_show("Grounding", 
                 "Please provide the bounding box coordinate of the region this sentence describes: the a black and white cat sitting on the edge of the bathtub", 
                 "assets/2.jpg")
        
    run_and_show("Object Detection", 
                 "Detect all objects in the provided image.", 
                 "assets/3.jpg")
        
    run_and_show("Semantic Segmentation", 
                 "Segment: poster, road, chandelier, swimming pool, arcade machine, armchair, floor, tray, sky, escalator, path, trade name, stove, canopy, sconce, stage, cradle, book, runway, signboard, pole, rug, lake, ship, pot, bulletin board, light, hill, barrel, vase, flag, blanket, clock, bicycle, sculpture, shower, bannister, stool, base, desk, wardrobe, bar, swivel chair, sea, table, painting, lamp, pool table, pillow, oven, animal, flower, mirror, food, sand, skyscraper, sidewalk, screen, windowpane, counter, hovel, refrigerator, cabinet, washer, mountain, waterfall, shelf, land, fountain, microwave, coffee table, radiator, truck, basket, dishwasher, bus, chair, grandstand, countertop, field, fireplace, boat, stairs, streetlight, sofa, cushion, building, towel, hood, plaything, plate, bed, ottoman, booth, rock, stairway, minibike, water, chest of drawers, car, glass, fan, screen door, plant, railing, curtain, toilet, computer, palm, river, television receiver, earth, tent, blind, box, bathtub, tank, seat, bed , bottle, pier, buffet, apparel, fence, tower, person, tree, kitchen island, grass, dirt track, sink, bookcase, ball, monitor, ceiling, door, column, van, crt screen, case, step, awning, bench, conveyer belt, traffic light, bridge, ashcan, wall, house, airplane, bag., without the background class.", 
                 "assets/4.jpg")
        
    run_and_show("Referring Segmentation", 
                 'Can you segment "hotdog on left" in this image?',
                 "assets/5.jpg")
        
    run_and_show("Open-Set Segmentation", 
                 "Segment all objects in this image.", 
                 "assets/6.jpg", 
                 seg_mode='openseg')
    
    # Openset needs to resize 2000 focal length with log dequantization
    run_and_show("Depth Estimation", 
                 "Estimate the depth.",
                 "assets/7.png")
    
    run_and_show("Pose Estimation", 
                 "Detect all persons and their poses from the image within the class set of MPII Human Pose Dataset. The output format should be a JSON-like string, containing person instances. Each person instance is enclosed in <person>...</person> tags. Within each person instance, provide the bounding box using <box>...</box> tags and their 16 keypoints using <kpt>...</kpt> tags. The bounding box is defined by <x_x1><y_y1><x_x2><y_y2> tags, and each keypoint is defined by <x_...><y_...><v_...> tags, where x, y are coordinates and v is visibility. The joints must be in this specific order: (0) right_ankle, (1) right_knee, (2) right_hip, (3) left_hip, (4) left_knee, (5) left_ankle, (6) pelvis, (7) thorax, (8) upper_neck, (9) head_top, (10) right_wrist, (11) right_elbow, (12) right_shoulder, (13) left_shoulder, (14) left_elbow, (15) left_wrist. Please output all detected persons with their bounding boxes and keypoints.",
                 "assets/8.png")