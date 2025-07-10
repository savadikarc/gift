QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"

# gsm8k_template = """Below is an instruction that \
# describes a task. Write a response that appropriately \
# completes the request.

# ### Instruction:
# %s

# ### Response: Let's think step by step.
# """

metamathqa_template = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""

no_header_prompt_template = """\
### Instruction:
%s

### Response:
"""

alpaca_prompt_template = """Below is an instruction that \
describes a task, paired with an input that provides \
further context. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

alpaca_prompt_no_input_template = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""

gsm8k_template = """%s\nAnswer the above question. First think step by step and then answer the final number.\n"""
