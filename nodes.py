import json
import time
import concurrent.futures
import openai
import comfy.utils  # ComfyUI的实用工具
import torch # 用于处理ComfyUI的IMAGE类型，虽然这里只是传递
from PIL import Image
import io
import base64
import re # Added for robust regex parsing of LLM output

# 定义一个自定义的类型，用于在节点之间传递配置信息
# 将 EX_LLM_SETTINGS 直接定义为一个字符串
EX_LLM_SETTINGS = "EX_LLM_SETTINGS"
# ComfyUI的BBOXES类型是预定义的，用于表示边界框列表
# ComfyUI的LIST类型是预定义的，用于表示Python列表

DEFAULT_PROMPT = '''## Role
You are a well-trained picture detector.Your task is to find the exact set of coordinates in the picture as required and return their locations in the form of coordinates,the detailed task steps is below,you MUST follow all the steps and give the right form as required.Remember:You don't get a second chance to correct the error, you have to output the correct content that fits the format all at once!

## Step 1:Analysis the prompt and separate them.

the user's postive prompt is "{objects}"
the user's negative prompt is "{negative_objects}"

You MUST analysis ALL the objects in both postive prompt and negative prompt.Extraction of information on all objects.  YOU MUST filter ALL the objects AND NEVER DISMISS ANY OBEJECT

Here are some examples,you must follow them:

"太阳和月亮" => “太阳,月亮”
"wood and grass"=> "wood,grass" 
"bread,apple" => "bread,apple" 
"苹果,梨" =>"苹果,梨"

Then store them into {{objects_list}} and {{negative_objects_list}}

## Step 2:Get the bboxes
Bboxes or bounding boxes is a kind of way to locate object in the image,the detect result should be like that:
"bbox_2d": [x1, y1, x2, y2]
Now,you need to detect all objects in {{objects_list}},and return the bbox_2d items,if there is only one object in {{objects_list}}, just store the result to {{bbox_result}},if there are two or more objects in {{objects_list}},store the result as the order,like:{{bbox_result0}},{{bbox_result1}},{{bbox_result2}}...

You need to obey every rules in the finding process as below:
1.Principles of exclusion
YOU MUST NOT include any objects when detecting objects in {{negative_objects_list}},not even one pixel.
2.Principles of minimisation
You MUST to get the minimum range in which the target can be detected in its entirety, and you HAVE TO make sure that there are no defects on the objects and they can't overlap each other., which are strictly forbidden.
3.Principle of non-omission
UNLESS there is more than a 98% probability of being sure that the object will not be found, YOU MUST find ALL the objects to be found and return their bbox values exactly as required

## Step 3:Formatted output
The format of output MUST be followed extremely, extremely strictly JSON.

If there is only one object {{objects_list}},just output:
```json
[
{{{bbox_result}}, "label": "{{objects_list[0]}}"}
]
```
If there are many results,just output:
```json
[
{{{bbox_result0}}, "label": "{{objects_list[0]}}"},
{{{bbox_result1}}, "label": "{{objects_list[1]}}"},
...
]
```
Objects must be arranged sequentially in order from top to bottom, left to right.

You MUST NOT make any adjustments to the output format, and you MUST replace the corresponding ‘{{}}’ element, you must output the JSON file directly, and you ARE NOT allowed to add any other content.

'''

# --------------------------------------------------------------------------------
# ExternalLLMDetectorSettings 节点
# 用于设置LLM请求的基本参数
# --------------------------------------------------------------------------------
class ExternalLLMDetectorSettings:
    """
    ComfyUI自定义节点: ExternalLLMDetectorSettings
    用于发起LLM的请求进行基本设置。
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入类型。
        """
        return {
            "required": {
                "base_url": ("STRING", {"default": "", "multiline": False}), # 请求的API地址
                "model_id": ("STRING", {"default": "", "multiline": False}), # 请求的模型名称
                "token": ("STRING", {"default": "", "multiline": False}),    # 用于请求鉴权的令牌 (对应OpenAI的api_key)
            }
        }

    RETURN_TYPES = (EX_LLM_SETTINGS,) # 定义节点的输出类型
    RETURN_NAMES = ("ex_llm_settings",) # 定义输出端口的名称
    FUNCTION = "do_settings" # 节点执行时调用的函数
    CATEGORY = "ExternalLLM" # 节点在ComfyUI UI中显示的类别

    def do_settings(self, base_url: str, model_id: str, token: str):
        """
        打包LLM连接设置。
        """
        # 将base_url, model_id, token打包成一个元组，作为自定义类型输出
        # ComfyUI会将这个元组作为一个整体传递给下一个节点
        settings_bundle = (base_url, model_id, token)
        print(f"[ExternalLLMDetectorSettings] LLM设置已打包: {settings_bundle}")
        return (settings_bundle,)


# --------------------------------------------------------------------------------
# ExternalLLMDetectorMainProcess 节点
# 用于处理发起LLM的请求，并发执行并保持顺序
# --------------------------------------------------------------------------------
class ExternalLLMDetectorMainProcess:
    """
    ComfyUI自定义节点: ExternalLLMDetectorMainProcess
    用于处理发起LLM的请求，支持并发和延迟。
    """
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入类型。
        """
        return {
            "required": {
                "ex_llm_settings": (EX_LLM_SETTINGS,), # 接受来自ExternalLLMDetectorSettings节点的打包设置
                "threads": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}), # 请求的并发数
                "delay": ("INT", {"default": 1, "min": 0, "max": 60, "step": 1}),   # 每次请求的延迟（秒）
                "images": ("IMAGE",), # 标准的 ComfyUI 图片组输入
                "objects": ("STRING", {"default": "human", "multiline": False}), # 用于替换prompt中的内容
                "negative_objects": ("STRING", {"default": "nothing", "multiline": False}), # 用于替换prompt中的内容
                "retries": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}), # LLM请求失败后的重试次数
                "prompt": ("STRING", {
                    "default": DEFAULT_PROMPT,
                    "multiline": True # 用户prompt，可多行输入
                }),
            }
        }
    RETURN_TYPES = ("IMAGE", "LIST") # 定义节点的输出类型
    RETURN_NAMES = ("images", "bboxes_strings_list") # 定义输出端口的名称
    FUNCTION = "process_llm_requests" # 节点执行时调用的函数
    CATEGORY = "ExternalLLM" # 节点在ComfyUI UI中显示的类别
    def process_llm_requests(self, ex_llm_settings: tuple, threads: int, delay: int, images: torch.Tensor, objects: str, negative_objects: str, prompt: str, retries: int):
        """
        解包LLM设置，并发对LLM发起请求，并按顺序收集结果。
        """
        base_url, model_id, token = ex_llm_settings # 解包设置
        # 初始化OpenAI客户端
        # 注意：这里的token对应OpenAI客户端的api_key参数
        try:
            client = openai.OpenAI(base_url=base_url, api_key=token)
        except Exception as e:
            print(f"[ExternalLLMDetectorMainProcess] 错误: 无法初始化OpenAI客户端。请检查base_url和token。错误详情: {e}")
            return (images, []) # 错误时返回原始图片和空列表
        # 用于存储按原始图像顺序排列的LLM响应字符串
        bboxes_strings_list = [None] * images.shape[0]
        # 准备并发任务：每个任务包含图像索引、填充后的prompt和对应的图片张量
        tasks = []
        for i in range(images.shape[0]):
            # 替换prompt中的占位符
            filled_prompt = prompt.replace("{objects}", objects)
            filled_prompt = filled_prompt.replace("{negative_objects}", negative_objects)
            tasks.append((i, filled_prompt, images[i])) # (图像索引, 完整的prompt, 单张图片张量)
        # 使用ThreadPoolExecutor进行并发请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            # 提交任务并存储future对象及其对应的原始图像索引
            futures = {executor.submit(self._call_llm_api, client, model_id, delay, retries, task[1], task[2]): task[0] for task in tasks}
            # 遍历已完成的future，并按原始顺序收集结果
            for future in concurrent.futures.as_completed(futures):
                original_index = futures[future] # 获取原始图像的索引
                try:
                    llm_response_content = future.result()
                    bboxes_strings_list[original_index] = llm_response_content
                    print(f"[ExternalLLMDetectorMainProcess] 图像 {original_index} 的LLM请求完成。")
                except openai.OpenAIError as e:
                    print(f"[ExternalLLMDetectorMainProcess] 图像 {original_index} 的LLM请求失败 (OpenAI Error): {e}")
                    bboxes_strings_list[original_index] = "" # 请求失败时存储空字符串
                except Exception as e:
                    print(f"[ExternalLLMDetectorMainProcess] 图像 {original_index} 的LLM请求失败 (未知错误): {e}")
                    bboxes_strings_list[original_index] = "" # 其它错误时存储空字符串
        print(f"[ExternalLLMDetectorMainProcess] 所有LLM请求处理完毕。")
        # 原封不动地返回原始图片，并返回LLM结果字符串列表
        return (images, bboxes_strings_list)
    def _call_llm_api(self, client: openai.OpenAI, model_id: str, delay: int, retries: int, prompt_text: str, image_tensor: torch.Tensor) -> str:
        """
        私有辅助方法：调用LLM API并处理响应。
        将图像张量转换为Base64编码的字符串，并将其与文本提示一起发送。
        """
        # 在每次请求前进行延迟
        if delay > 0:
            time.sleep(delay)
        print(f"[ExternalLLMDetectorMainProcess] 发起LLM请求 (模型: {model_id})...")
        # 1. 将torch.Tensor图像转换为PIL Image
        # ComfyUI的图像张量通常是 (H, W, C) 范围0-1的浮点数
        image_np = image_tensor.cpu().numpy()
        image_pil = Image.fromarray((image_np * 255).astype('uint8'))
        # 2. 将PIL Image转换为Base64编码的JPEG字符串
        buffered = io.BytesIO()
        image_pil.save(buffered, format="JPEG") # 可以选择PNG，但JPEG通常更小
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # 构造多模态消息内容
        messages_content = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
        for attempt in range(retries + 1):
            try:
                chat_completion = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "user", "content": messages_content} # 包含文本和图像内容
                    ]
                )
                # 提取LLM的响应内容
                response_content = chat_completion.choices[0].message.content
                return response_content
            except openai.OpenAIError as e:
                if attempt < retries:
                    print(f"[ExternalLLMDetectorMainProcess] LLM请求失败 (尝试 {attempt + 1}/{retries + 1}): {e}. 正在重试...")
                    time.sleep(1) # 简单的重试延迟
                else:
                    print(f"[ExternalLLMDetectorMainProcess] LLM请求在 {retries + 1} 次尝试后仍然失败: {e}. 返回空内容。")
                    return "" # 最后一次尝试失败，返回空内容
            except Exception as e:
                if attempt < retries:
                    print(f"[ExternalLLMDetectorMainProcess] LLM请求失败 (尝试 {attempt + 1}/{retries + 1}, 未知错误): {e}. 正在重试...")
                    time.sleep(1) # 简单的重试延迟
                else:
                    print(f"[ExternalLLMDetectorMainProcess] LLM请求在 {retries + 1} 次尝试后仍然失败 (未知错误): {e}. 返回空内容。")
                    return "" # 最后一次尝试失败，返回空内容


# --------------------------------------------------------------------------------
# ExternalLLMDetectorBboxesConvert 节点
# 用于将LLM返回的BboxesStrings转换为符合SAM2格式的bboxes
# --------------------------------------------------------------------------------
class ExternalLLMDetectorBboxesConvert:
    """
    ComfyUI自定义节点: ExternalLLMDetectorBboxesConvert
    用于将LLM返回的bboxes_strings_list转换为符合SAM2格式的bboxes。
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入类型。
        """
        return {
            "required": {
                "bboxes_strings_list": ("LIST",), # 接受来自ExternalLLMDetectorMainProcess节点的LLM结果字符串列表
            }
        }

    RETURN_TYPES = ("BBOXES",) # 定义节点的输出类型
    RETURN_NAMES = ("sam2_bboxes",) # 定义输出端口的名称
    FUNCTION = "convert_bboxes" # 节点执行时调用的函数
    CATEGORY = "ExternalLLM" # 节点在ComfyUI UI中显示的类别

    def convert_bboxes(self, bboxes_strings_list: list) -> tuple:
        """
        将LLM返回的JSON字符串列表解析并转换为SAM2兼容的边界框格式。
        SAM2格式示例:
        [
            [[10, 10, 100, 100], [200, 200, 300, 300]],  # 图像1，两个边界框
            [[50, 50, 150, 150]],                        # 图像2，一个边界框
            [[100, 100, 200, 200], [300, 300, 400, 400], [500, 500, 600, 600]] # 图像3，三个边界框
        ]
        """
        sam2_bboxes_output = [] # 最终的SAM2格式边界框列表
        print(f"[ExternalLLMDetectorBboxesConvert] 开始处理LLM回应的bbox字符串，原始字符串: {bboxes_strings_list}")
        for i, bbox_str in enumerate(bboxes_strings_list):
            current_image_bboxes = [] # 当前图像的边界框列表
            if not bbox_str:
                print(f"[ExternalLLMDetectorBboxesConvert] 警告: 图像 {i} 的bbox字符串为空或None，跳过。")
                sam2_bboxes_output.append([]) # 为当前图像添加空列表
                continue

            # --- ROBUST PARSING LOGIC ---
            extracted_bboxes_from_regex = []
            # Regex to find all occurrences of "bbox_2d": [x, y, w, h]
            # This regex is designed to be robust to whitespace and capture the four integer coordinates.
            bbox_regex = r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
            matches = re.findall(bbox_regex, bbox_str)

            for match in matches:
                try:
                    # Convert captured strings to integers
                    bbox = [int(coord) for coord in match]
                    extracted_bboxes_from_regex.append(bbox)
                except ValueError:
                    print(f"[ExternalLLMDetectorBboxesConvert] 警告: 图像 {i} 的bbox regex匹配到非整数坐标。跳过该项。匹配: {match}")

            if extracted_bboxes_from_regex:
                print(f"[ExternalLLMDetectorBboxesConvert] 提示: 图像 {i} 成功通过正则表达式提取到边界框。")
                current_image_bboxes.extend(extracted_bboxes_from_regex)
            else:
                # Fallback to JSON parsing if regex didn't find anything or failed
                try:
                    parsed_json = json.loads(bbox_str)

                    # Existing logic for handling parsed_json
                    final_bbox_items = []

                    # Prioritize handling the problematic nested format: {'bbox_2d': [{'bbox_2d': [coords], 'label': '...'}, ...]}
                    if isinstance(parsed_json, dict) and "bbox_2d" in parsed_json and \
                       isinstance(parsed_json["bbox_2d"], list) and \
                       len(parsed_json["bbox_2d"]) > 0 and \
                       isinstance(parsed_json["bbox_2d"][0], dict) and \
                       "bbox_2d" in parsed_json["bbox_2d"][0]:
                        print(f"[ExternalLLMDetectorBboxesConvert] 提示: 图像 {i} 的JSON解析结果是包含嵌套bbox字典列表的字典，正在提取嵌套列表。")
                        final_bbox_items = parsed_json["bbox_2d"]
                    elif isinstance(parsed_json, list):
                        # Case 1: LLM returned a list of bbox dictionaries directly
                        final_bbox_items = parsed_json
                    elif isinstance(parsed_json, dict):
                        # Check for {'bbox_2d': [list of coordinates]} format
                        if "bbox_2d" in parsed_json and \
                             isinstance(parsed_json["bbox_2d"], list) and \
                             len(parsed_json["bbox_2d"]) > 0 and \
                             isinstance(parsed_json["bbox_2d"][0], list) and \
                             len(parsed_json["bbox_2d"][0]) == 4 and \
                             all(isinstance(coord, (int, float)) for coord in parsed_json["bbox_2d"][0]):
                            print(f"[ExternalLLMDetectorBboxesConvert] 提示: 图像 {i} 的JSON解析结果是包含嵌套坐标列表的字典，正在转换为bbox字典列表。")
                            final_bbox_items = [{"bbox_2d": coords} for coords in parsed_json["bbox_2d"]]
                        # Check for single bbox dictionary format (e.g., LLM returns a single bbox dict, not a list)
                        elif "bbox_2d" in parsed_json and \
                             isinstance(parsed_json["bbox_2d"], list) and \
                             len(parsed_json["bbox_2d"]) == 4 and \
                             all(isinstance(coord, (int, float)) for coord in parsed_json["bbox_2d"]):
                            print(f"[ExternalLLMDetectorBboxesConvert] 提示: 图像 {i} 的JSON解析结果是单个边界框字典，将其视为包含一个边界框的列表。")
                            final_bbox_items = [parsed_json]
                        else:
                            # Unrecognized dictionary format, or empty list
                            print(f"[ExternalLLMDetectorBboxesConvert] 警告: 图像 {i} 的JSON解析结果是无法识别的字典格式，跳过。数据: {parsed_json}")
                            sam2_bboxes_output.append([])
                            continue
                    else:
                        # Invalid top-level format
                        print(f"[ExternalLLMDetectorBboxesConvert] 警告: 图像 {i} 的JSON解析结果既不是列表也不是字典，跳过。原始字符串: {bbox_str}")
                        sam2_bboxes_output.append([])
                        continue

                    # Now, iterate through final_bbox_items, which should always be a list of bbox dictionaries
                    for item in final_bbox_items:
                        # Check if it contains "bbox_2d" key and its value is a list of 4 numbers
                        if isinstance(item, dict) and "bbox_2d" in item and \
                           isinstance(item["bbox_2d"], list) and \
                           len(item["bbox_2d"]) == 4 and \
                           all(isinstance(coord, (int, float)) for coord in item["bbox_2d"]):
                            # Convert coordinates to integers (ComfyUI's BBOXES usually expects integers)
                            bbox = [int(coord) for coord in item["bbox_2d"]]
                            current_image_bboxes.append(bbox)
                        else:
                            print(f"[ExternalLLMDetectorBboxesConvert] 警告: 图像 {i} 中发现无效的bbox格式或缺少'bbox_2d'键。跳过该项。数据: {item}")

                except json.JSONDecodeError:
                    # If direct JSON parsing fails, and regex also failed, try the old "wrap in []" logic
                    if bbox_str.strip().startswith('{') and '},{' in bbox_str:
                        try:
                            parsed_json = json.loads(f"[{bbox_str.strip()}]")
                            print(f"[ExternalLLMDetectorBboxesConvert] 提示: 图像 {i} 的JSON字符串被识别为多个对象，已尝试添加外部列表括号并成功解析。")
                            # Process the newly parsed_json (which should be a list)
                            if isinstance(parsed_json, list):
                                for item in parsed_json:
                                    if isinstance(item, dict) and "bbox_2d" in item and \
                                       isinstance(item["bbox_2d"], list) and \
                                       len(item["bbox_2d"]) == 4 and \
                                       all(isinstance(coord, (int, float)) for coord in item["bbox_2d"]):
                                        bbox = [int(coord) for coord in item["bbox_2d"]]
                                        current_image_bboxes.append(bbox)
                                    else:
                                        print(f"[ExternalLLMDetectorBboxesConvert] 警告: 图像 {i} 中发现无效的bbox格式或缺少'bbox_2d'键在重试解析后。跳过该项。数据: {item}")
                            else:
                                print(f"[ExternalLLMDetectorBboxesConvert] 错误: 图像 {i} 的JSON字符串尝试添加外部列表括号后解析结果不是列表。原始字符串: {bbox_str}")
                                sam2_bboxes_output.append([])
                                continue
                        except json.JSONDecodeError as e_retry:
                            print(f"[ExternalLLMDetectorBboxesConvert] 错误: 图像 {i} 的JSON字符串尝试添加外部列表括号后仍解析失败。错误: {e_retry}。原始字符串: {bbox_str}")
                            sam2_bboxes_output.append([])
                            continue
                    else:
                        print(f"[ExternalLLMDetectorBboxesConvert] 错误: 图像 {i} 的JSON字符串解析失败。原始字符串: {bbox_str}")
                        sam2_bboxes_output.append([])
                        continue
                except Exception as e:
                    print(f"[ExternalLLMDetectorBboxesConvert] 错误: 处理图像 {i} 的bbox时发生未知错误。错误: {e}。原始字符串: {bbox_str}")
                    sam2_bboxes_output.append([])
                    continue

            sam2_bboxes_output.append(current_image_bboxes)
        print(f"[ExternalLLMDetectorBboxesConvert] Bboxes已转换为SAM2格式。处理了 {len(sam2_bboxes_output)} 张图片。")
        print(f'[ExternalLLMDetectorBboxesConvert] 处理后的Bboxes为：{sam2_bboxes_output,}')
        return (sam2_bboxes_output,)


# --------------------------------------------------------------------------------
# ComfyUI 节点映射
# --------------------------------------------------------------------------------
# 将节点类映射到ComfyUI的内部系统，使其可以在UI中显示和使用
NODE_CLASS_MAPPINGS = {
    "ExternalLLMDetectorSettings": ExternalLLMDetectorSettings,
    "ExternalLLMDetectorMainProcess": ExternalLLMDetectorMainProcess,
    "ExternalLLMDetectorBboxesConvert": ExternalLLMDetectorBboxesConvert,
}

# 为节点提供更友好的显示名称，将在ComfyUI的节点搜索菜单中显示
NODE_DISPLAY_NAME_MAPPINGS = {
    "ExternalLLMDetectorSettings": "ExternalLLMDetectorSettings",
    "ExternalLLMDetectorMainProcess": "ExternalLLMDetectorMainProcess",
    "ExternalLLMDetectorBboxesConvert": "ExternalLLMDetectorBboxesConvert",
}
