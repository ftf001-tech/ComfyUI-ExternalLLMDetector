import json
import time
import concurrent.futures
import openai
import comfy.utils  # ComfyUI的实用工具
import torch # 用于处理ComfyUI的IMAGE类型，虽然这里只是传递
from PIL import Image
import io
import base64

# 定义一个自定义的类型，用于在节点之间传递配置信息
# 将 EX_LLM_SETTINGS 直接定义为一个字符串
EX_LLM_SETTINGS = "EX_LLM_SETTINGS"
# ComfyUI的BBOXES类型是预定义的，用于表示边界框列表
# ComfyUI的LIST类型是预定义的，用于表示Python列表


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
                "object": ("STRING", {"default": "human", "multiline": False}), # 用于替换prompt中的内容
                "prompt": ("STRING", {
                    "default": "Locate the {objects} and output bbox in JSON.The format must looks like:\n```json\n[\n{\"bbox_2d\": [123, 456, 789, 012], \"label\": \"target_object\"}\n]\n```",
                    "multiline": True # 用户prompt，可多行输入
                }),
            }
        }
    RETURN_TYPES = ("IMAGE", "LIST") # 定义节点的输出类型
    RETURN_NAMES = ("images", "bboxes_strings_list") # 定义输出端口的名称
    FUNCTION = "process_llm_requests" # 节点执行时调用的函数
    CATEGORY = "ExternalLLM" # 节点在ComfyUI UI中显示的类别
    def process_llm_requests(self, ex_llm_settings: tuple, threads: int, delay: int, images: torch.Tensor, object: str, prompt: str):
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
            filled_prompt = prompt.replace("{objects}", object)
            tasks.append((i, filled_prompt, images[i])) # (图像索引, 完整的prompt, 单张图片张量)
        # 使用ThreadPoolExecutor进行并发请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            # 提交任务并存储future对象及其对应的原始图像索引
            futures = {executor.submit(self._call_llm_api, client, model_id, delay, task[1], task[2]): task[0] for task in tasks}
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
    def _call_llm_api(self, client: openai.OpenAI, model_id: str, delay: int, prompt_text: str, image_tensor: torch.Tensor) -> str:
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
        chat_completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": messages_content} # 包含文本和图像内容
            ],
            # 关键：要求LLM返回JSON格式
            response_format={"type": "json_object"}
        )
        # 提取LLM的响应内容
        response_content = chat_completion.choices[0].message.content
        return response_content


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
        for i, bbox_str in enumerate(bboxes_strings_list):
            current_image_bboxes = [] # 当前图像的边界框列表
            if not bbox_str:
                print(f"[ExternalLLMDetectorBboxesConvert] 警告: 图像 {i} 的bbox字符串为空或None，跳过。")
                sam2_bboxes_output.append([]) # 为当前图像添加空列表
                continue
            try:
                # 尝试解析JSON字符串
                parsed_json = json.loads(bbox_str)
                # 根据JSON解析结果的类型进行处理
                bbox_data_list = []
                if isinstance(parsed_json, list):
                    # 如果是列表，直接使用它
                    bbox_data_list = parsed_json
                elif isinstance(parsed_json, dict):
                    # 如果是单个字典，将其包装成一个包含该字典的列表
                    print(f"[ExternalLLMDetectorBboxesConvert] 提示: 图像 {i} 的JSON解析结果是单个字典，将其视为包含一个边界框的列表。")
                    bbox_data_list = [parsed_json]
                else:
                    # 如果既不是列表也不是字典，则视为无效格式
                    print(f"[ExternalLLMDetectorBboxesConvert] 警告: 图像 {i} 的JSON解析结果既不是列表也不是字典，跳过。原始字符串: {bbox_str}")
                    sam2_bboxes_output.append([])
                    continue # 跳过当前图像的处理，进入下一张
                # 遍历处理每个边界框对象（无论是原始列表还是包装后的列表）
                for item in bbox_data_list:
                    # 检查是否包含"bbox_2d"键且其值为包含4个数字的列表
                    if isinstance(item, dict) and "bbox_2d" in item and \
                       isinstance(item["bbox_2d"], list) and \
                       len(item["bbox_2d"]) == 4 and \
                       all(isinstance(coord, (int, float)) for coord in item["bbox_2d"]):
                        # 将坐标转换为整数（ComfyUI的BBOXES通常期望整数）
                        bbox = [int(coord) for coord in item["bbox_2d"]]
                        current_image_bboxes.append(bbox)
                    else:
                        print(f"[ExternalLLMDetectorBboxesConvert] 警告: 图像 {i} 中发现无效的bbox格式或缺少'bbox_2d'键。跳过该项。数据: {item}")
            except json.JSONDecodeError as e:
                print(f"[ExternalLLMDetectorBboxesConvert] 错误: 图像 {i} 的JSON字符串解析失败。错误: {e}。原始字符串: {bbox_str}")
                # 解析失败时，为当前图像添加空列表
                current_image_bboxes = []
            except Exception as e:
                print(f"[ExternalLLMDetectorBboxesConvert] 错误: 处理图像 {i} 的bbox时发生未知错误。错误: {e}。原始字符串: {bbox_str}")
                current_image_bboxes = []
            sam2_bboxes_output.append(current_image_bboxes)
        print(f"[ExternalLLMDetectorBboxesConvert] Bboxes已转换为SAM2格式。处理了 {len(sam2_bboxes_output)} 张图片。")
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

