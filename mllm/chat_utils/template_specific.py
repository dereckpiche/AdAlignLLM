"""
File: mllm/chat_utils/template_specific.py
Summary: Stores chat template variants and assistant postfix tensors per tokenizer.
"""

import huggingface_hub
import torch
from transformers import AutoTokenizer

custom_llama3_template = """
{%- if add_system_prompt %}
    {{- '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>' }}
{%- endif %}
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
{%- endfor %}

{%- if add_generation_prompt %}
    {{- '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}
{%- endif %}
"""

qwen2_assistant_postfix = (
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    .encode("\n", return_tensors="pt")
    .flatten()
)
qwen3_assistant_postfix = (
    AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    .encode("\n", return_tensors="pt")
    .flatten()
)
gemma3_assistant_postfix = (
    AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    .encode("\n", return_tensors="pt")
    .flatten()
)
custom_qwen2_template = """
{%- if add_system_prompt %}
    {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if reasoning_content %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""

custom_qwen3_template = """
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
"""

custom_gemma3_template = """
{%- if add_system_prompt %}
{{- bos_token -}}
{%- endif %}
{%- for message in messages -%}
{%- if message['role'] == 'assistant' -%}
{%- set role = 'model' -%}
{%- else -%}
{%- set role = message['role'] -%}
{%- endif -%}
{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{ '<start_of_turn>model\n' }}
{%- endif -%}
"""
