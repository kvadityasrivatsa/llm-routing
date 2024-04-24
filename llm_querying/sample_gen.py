import os
import json
import pickle
import argparse
from glob import glob
from tqdm import tqdm
from copy import deepcopy

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_PATHS = {
    "llama2-7b-lm":     "/l/users/name/huggingface/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852",
    "llama2-7b-chat":   "/l/users/name/huggingface/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852",
    "llama2-13b-lm":    "/l/users/name/huggingface/models--meta-llama--Llama-2-13b-hf/snapshots/dc1d3b3bfdb69df26f8fc966c16353274b138c55",
    "llama2-13b-chat":  "/l/users/name/huggingface/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496",
    "llama2-70b-chat":  "/l/users/name/huggingface/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e1ce257bd76895e0864f3b4d6c7ed3c4cdec93e2",
    "mistral-7b-inst":  "/l/users/name/huggingface/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/d2953fb360de291c66caf55739473ca4b72eca0b",
    "mistral-7b-lm":    "/l/users/name/huggingface/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24",
    "qwen-14b":         "/l/users/name/huggingface/models--Qwen--Qwen-14B-Chat/snapshots/27ce7630195e625dce76050f5f982591bc889c24",
    "tigerbot-70b":     "/l/users/name/huggingface/models--TigerResearch--tigerbot-70b-chat-v3/snapshots/ef894f53efe7cc8b6f5ab3e7ab48279d802c44c9",
    "metamath-7b":     "/root/sample_gen/model/models--meta-math--MetaMath-7B-V1.0/snapshots/e634cdc03e2e32e1ad35ea1aa2418090221a0951",
    "vicuna-13b":       "/l/users/name/huggingface/models--lmsys--vicuna-13b-v1.5/snapshots/3deb0106f72a3a433f0c6ea0cb978bdf14bcd3a6",
    "falcon-7b":        "/l/users/name/huggingface/models--tiiuae--falcon-7b/snapshots/898df1396f35e447d5fe44e0a3ccaaaa69f30d36",
    "gemma-7b-it":      "/root/sample_gen/model/models--google--gemma-7b-it/snapshots/dec4b13d574762bd36f0a1b75541439bd852b2e8",
    "gemma-7b":         "/root/sample_gen/model/models--google--gemma-7b/snapshots/7646584ed746494da9e1058b1be53d1be8b2ee73",
    "gemma-2b":         "/root/sample_gen/hf_models/models--google--gemma-2b/snapshots/9d067f00def958594aaa16b39a65b07d69ca655b",
    "gemma-2b-it":      "/root/sample_gen/hf_models/models--google--gemma-2b-it/snapshots/718cb189da9c5b2e55abe86f2eeffee9b4ae0dad",
    "qwen2-7b-chat":    "/l/users/name/huggingface/models--Qwen--Qwen1.5-7B-Chat/snapshots/03df580367e73ba602b3b678fbdf650fa3593e89",
}

MODEL_HGF_KEY = {
    "llama2-7b-lm":     "meta-llama/Llama-2-7b-hf",
    "llama2-7b-chat":   "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat":  "meta-llama/Llama-2-13b-chat-hf",
    "llama2-13b-lm":    "meta-llama/Llama-2-13b-hf",
    "llama2-70b-chat":  "meta-llama/Llama-2-70b-chat-hf",
    "mistral-7b-inst":  "mistralai/Mistral-7B-Instruct-v0.1",
    "mistral-7b-lm":    "mistralai/Mistral-7B-v0.1",
    "metamath-7b":     "meta-math/MetaMath-7B-V1.0",
    "vicuna-13b":       "lmsys/vicuna-13b-v1.5",
    "falcon-7b":        "tiiuae/falcon-7b",
    "gemma-7b-it":      "google/gemma-7b-it",
    "gemma-7b":         "google/gemma-7b",
    "gemma-2b":         "google/gemma-2b",
    "gemma-2b-it":      "google/gemma-2b-it",
    "qwen2-7b-chat":    "Qwen/Qwen1.5-7B-Chat",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model and query paths")
    parser.add_argument("--batch-size",         type=int, default=1, help="Query batch size.")
    parser.add_argument("--chat-mode",          type=int, default=1, help="Chat Mode. Set to 1 if you wich to use the transformers' tokenizer driven chat template, else set to 0.")
    parser.add_argument("--query-field",        type=str, default="to_be_queried", help="Specify the name of the field (dict key) to be queried.")
    parser.add_argument("--query-limit",        type=int, default=-1, help="Maximum number of queries to process. If -1, will assume the length of JSON queries.")
    parser.add_argument("--max-len",            type=int, default=2000, help="Maximum overall length of text. Includes prompt length.")
    parser.add_argument("--model-name",         type=str, choices=MODEL_PATHS.keys(), help=f"Pick any one of the following LLMs: {list(MODEL_PATHS.keys())}.")
    parser.add_argument("--n-seq",              type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--query-paths",        type=str, nargs="+", default=[], help="List of paths to JSON files to query. JSON is expected to be a list of dictionaries, with each dict conatining the relevant field to be queried.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty. Not linear.")
    parser.add_argument("--seed",               type=int, default=0, help="Generation seed.")
    parser.add_argument("--temperature",        type=float, default=0.7, help="Generation temperature. Set a higher temperature for more randomized token-selection.")
    parser.add_argument("--top-k",              type=int, default=-1, help="Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens")
    parser.add_argument("--uuid",               type=str, default="UUID", help="Universally unique Identifier to ensure multiple runs are distinguishable.")
    parser.add_argument("--inference_mode",     type=str, default="inference_mode", help="To enable zero-shot or few-shot.")

    args = parser.parse_args()
    print( "Loaded Model Aguments :", vars(args))
    data_name = args.query_paths[0].strip().split('/')[-1].split('.')[0]
    
    addn_args = {}

    os.makedirs("/root/sample_gen/outputs", exist_ok=True)
    addn_args["target_dir"] = "/root/sample_gen/outputs/"

    model_path = MODEL_PATHS[args.model_name]

    llm = LLM(
        model = model_path, 
        tensor_parallel_size = torch.cuda.device_count(),
        seed = args.seed,
        tokenizer_mode='auto',
        trust_remote_code=True
    )

    print(f"The {args.model_name} model is loaded with vLLM from Path: { model_path}")
    # Add more as needed from https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    sampling_params = SamplingParams(
        n=2,
        top_k=args.top_k,
        temperature=args.temperature, 
        repetition_penalty=args.repetition_penalty, 
        max_tokens=args.max_len,
        stop="<END_ANSWER>"
    )

    def add_question_to_prompt_template(q):
        out = q['0']+" "
        for bullet, option in zip(["A","B","C","D"],['1','2','3','4']):
            out += f"({bullet}) {q[option]} "
        out += "\n"
        temp_q =out.replace('\n', '').replace('\n\n', '').replace('\n\n\n', '')
        new_template= f"Question: {temp_q.strip()} "
        return new_template
    
    def zero_shot_prompt_formatting(data_temp, data_name, chat_mode, model_name):
        # Zero-shot COT prompt options:
        #zcot = "Answer: Let's think step by step."
        #zcot = "Answer: Let's think step by step. Be concise in your answer. Clearly mention a single answer in the end."
        #zcot = "Answer: Let's think step by step. Be concise in your response. Clearly mention a single answer at the end response after ###."
        
        for d in data_temp:
            if data_name == 'gsm8k':
                ques = d["question"]
                query = f" Question: {ques.strip()}"
            elif data_name =='mmlu-2':
                query = add_question_to_prompt_template(d['raw'])
                
            if chat_mode:
                if "llama2" in model_name:
                    chat = [{"role": "system", "content": "You are a helpful assistant. Let's think and answer step by step for the below question. Clearly mention single answer in the end."},
                             {"role": "user", "content": '"'+str(query)+'"'}]
                else:
                    chat = [{"role": "user", "content": "You are a helpful assistant. Let's think and answer step by step for the below question. Clearly mention single answer in the end." + "\n"+'"'+str(query)+'"'}] 
                tokenizer = AutoTokenizer.from_pretrained(MODEL_HGF_KEY[model_name], trust_remote_code=True)
                d['zprompt'] = tokenizer.apply_chat_template(chat, tokenize=False)
            else:
                d['zprompt'] = +str(query)+" "+str(zcot)
        return data_temp
    
    def few_shot_prompt_formatting(data_temp, data_name, chat_mode, model_name):
        # Few-shot COT prompt options:
        fcot = "<START_QUESTION> Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to  study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day? <END_QUESTION> \n <START_ANSWER> Answer: Let's think step by step. Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total. For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total. Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days. However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks. They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes. And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours. So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total. They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75 They will need to plan to study 4 days to allow for all the time they need. The answer is 4. <END_ANSWER> \n <START_QUESTION> Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together? <END_QUESTION> \n <START_ANSWER> Answer: Let's think step by step. Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers. His team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers. They scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws. All together his team scored 50+24+10= 84 points Mark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers. His opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers. They also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws. All together Mark's opponents scored 100+12+5=117 points The total score for the game is both team's scores added together, so it is 84+117=201 points. The answer is 201. <END_ANSWER> \n <START_QUESTION> Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles? <END_QUESTION> \n <START_ANSWER> Answer: Let's think step by step. When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24. The total number of marbles she'll have is 60+24 = 84. If Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees. If Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees. The total number of frisbees she'll have will increase to 30+12 = 42. Bella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards. If she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards. The total number of deck cards she'll have is 10+4 = 14. Together, Bella will have a total of 14+42+84 = 140 items. The answer is 140. <END_ANSWER> \n <START_QUESTION> Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there? <END_QUESTION> \n <START_ANSWER> Answer: Let's think step by step. For the first three baskets, the number of apples and oranges in one basket is 9+15=24. In total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets. Since there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets. The number of apples in the fourth basket is 9-2=7. There are also 15-2=13 oranges in the fourth basket. The combined number of oranges and apples in the fourth basket is 13+7=20. The fourth basket also contains 14-2=12 bananas. In total, the fourth basket has 20+12=32 fruits. The four baskets together have 32+114=146 fruits. The answer is 146. <END_ANSWER> \n <START_QUESTION> Question: Question: You can buy 4 apples or 1 watermelon for the same price. You bought 36 fruits evenly split between oranges, apples and watermelons, and the price of 1 orange is $0.50. How much does 1 apple cost if your total bill was $66? <END_QUESTION> \n <START_ANSWER> Answer: Let's think step by step. If 36 fruits were evenly split between 3 types of fruits, then I bought 36/3 = 12 units of each fruit. If 1 orange costs $0.50 then 12 oranges will cost $0.50 * 12 = $6. If my total bill was $66 and I spent $6 on oranges then I spent $66 - $6 = $60 on the other 2 fruit types. Assuming the price of watermelon is W, and knowing that you can buy 4 apples for the same price and that the price of one apple is A, then 1W=4A. If we know we bought 12 watermelons and 12 apples for $60, then we know that $60 = 12W + 12A. Knowing that 1W=4A, then we can convert the above to $60 = 12(4A) + 12A $60 = 48A + 12A $60 = 60A. Then we know the price of one apple (A) is $60/60= $1. The answer is 1. <END_ANSWER>"
        
        for d in data_temp:
            if data_name == 'gsm8k':
                if chat_mode:
                    if "llama2" in model_name:
                        chat = [{"role": "system", "content": "You are a helpful assistant."},
                                 {"role": "user", "content": str(fcot)+" \n <START_QUESTION> Question: "+str(d["question"])+" <END_QUESTION> \n <START_ANSWER> Answer:"}]
                    else:
                        chat = [{"role": "user", "content": "You are a helpful assistant." + "\n\n" + str(fcot)+" \n <START_QUESTION> Question: "+str(d["question"])+" <END_QUESTION> \n <START_ANSWER> Answer:"}] 
                        tokenizer = AutoTokenizer.from_pretrained(MODEL_HGF_KEY[model_name], trust_remote_code=True)
                        d['zprompt'] = tokenizer.apply_chat_template(chat, tokenize=False)
                else:
                    d['zprompt'] = str(fcot)+" \n <START_QUESTION> Question: "+str(d["question"])+" <END_QUESTION> \n <START_ANSWER> Answer:"
                    #d['zprompt'] = str(fcot)+" \n <START_QUESTION> Question: "+str(d["question"])+" <END_QUESTION> \n <START_ANSWER> Answer: Let's think step by step."
            if data_name =='mmlu-2':
                if chat_mode:
                    if "llama2" in model_name:
                        chat = [{"role": "system", "content": "You are a helpful assistant."},
                                 {"role": "user", "content": d["few_shot_cot_prompt"].strip()}]
                    else:
                        chat = [{"role": "user", "content": "You are a helpful assistant." + "\n\n" + d["few_shot_cot_prompt"].strip()}] 
                        tokenizer = AutoTokenizer.from_pretrained(MODEL_HGF_KEY[model_name], trust_remote_code=True)
                        d['zprompt'] = tokenizer.apply_chat_template(chat, tokenize=False)
                else:
                    d['zprompt'] = d["few_shot_cot_prompt"].strip()
        return data_temp
    
    for fpath in tqdm(args.query_paths):
        print("Loaded Data Path:", fpath)
        addn_args['prefix'] = f"{fpath.split('/')[-1].split('.')[0]}_{args.model_name}_temp{args.temperature}_{args.uuid}_{args.inference_mode}_chatmode_{args.chat_mode}"

        data = json.load(open(fpath,'r'))

        # data cleaning
        for d in data:
            if data_name == 'gsm8k':
                del d["few_shot_cot_prompt"]
                del d["zero_shot_cot_prompt"]
        print("Original Total Number of Prompts:", len(data))
        
        #Query filtering 
        if args.query_limit != -1:
            if args.query_limit < 1:
                raise Exception("Invalid query limit. Expected >=1.")
            data = data[:args.query_limit]
        print("Total Number of Prompts Queried:", len(data))
        
        if args.inference_mode == "zero-shot":
            # Prompt formatting zero-shot
            final_data = zero_shot_prompt_formatting(data, data_name, args.chat_mode, args.model_name)
        else:        
            # Prompt formatting few-shot
            final_data = few_shot_prompt_formatting(data, data_name, args.chat_mode, args.model_name)
        
        assert len(data) == len(final_data)
        
        prompts = [pmpt["zprompt"] for pmpt in final_data]

        #prompt extension
        extended_prompts = []
        for p in prompts:
            extended_prompts.extend([p for _ in range(args.n_seq)])
        print("Extended Number of Queried Prompts:", len(extended_prompts))
          
        
        print("Response Generation Started ..........................")
        responses = [res for res in llm.generate(extended_prompts, sampling_params, use_tqdm=True)]
        print("Response Generation Completed!!!")        
        
        collect_res = {}
        for res in responses:
            out_prompt = str(res.prompt.strip())
            out_response = str(res.outputs[0].text.strip())
            if out_prompt not in collect_res:
                collect_res[out_prompt] = [out_response]
            else:
                collect_res[out_prompt].append(out_response)
            
        for d in final_data:
            if str(d["zprompt"].strip()) in collect_res:
                d["query_response"] = [out for out in collect_res[str(d["zprompt"].strip())]]
        json.dump(
            final_data,
            open(os.path.join(addn_args["target_dir"],f"{addn_args['prefix']}.json"), "w"),
            indent=3,
        )
