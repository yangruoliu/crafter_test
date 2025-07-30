def compose_planning_prompt(rules):

    PLANNING_PROMPT = "You are playing an open world survival game. Below are some of the rules of the game.\n" + rules + "\nInorder to achieve the current goal, you need to propose a TODO list. The size of the TODO list should be strictly less then 6. Your response should be a python list. For example:\n "

    PLANNING_PROMPT += """

    ["subgoal1", "subgoal2", "subgoal3", "subgoal4", "current_goal"]

    Now consider:

    """

    return PLANNING_PROMPT

def compose_llm_agent_prompt(rules, model_description, current_goal, info, last_model_call=""):

    LLM_AGENT_PROMPT = "You are playing an open world survival game. Below are some of the rules of the game.\n\n" + rules + "\n\nThe pre-trained RL models below are available:\n\n" + model_description + "\n\nYou can and only can use those models to achieve the following goal\n\nCuurent goal:" + current_goal + "Here is your observation:\n\n" + str(info['obs']) + "\n\n" + str(info['history']) + "\n\n Here is your achievements where 1 stand for what you have already reached. Make sure don't reach achievements repeatly unless nessesary.\n\n" + str(info['achievements']) + "\n\nYour last model call is: " + last_model_call + "\n\nYour answer should strictly follow the template below and NOTHING ELSE:\n\n{'choice':'call TODO1', 'reason': 'TODO2'}\n\n 'TODO' should be one of the models in `model_description` and TODO2 is your explanation. Your should list the rules your used in your explanation"

    return LLM_AGENT_PROMPT

def compose_llm_attention_prompt(rules, obs, goal):

    LLM_ATTENTION_PROMPT = '''You are an intelligent assistant helping an agent navigate a 2D open-world game. The agent receives observations in natural language and must decide what areas in the environment to focus on.

Your task is: Given the agent's observation, output a JSON object containing a list of attention targets. Each target includes:
- "object": the name of an object seen in the environment
- "direction": the compass direction (choose from "north", "west", "east", "south", "south-east", "north-east", "south-west", "north-west")
- "distance": integer number of steps
- "priority": a float between 0 and 1 indicating how important this object is to the agent's next action

Here are the rules of the game:

''' + rules + '''

For example, 

Observation:

You see:

- water 2 steps to your south
- grass 1 steps to your north
- stone 6 steps to your north-west
- tree 4 steps to your north-east
- lava 1 steps to your west 
- coal 7 steps to your north-west

Your status:
- health: 0/9
- food: 5/9
- drink: 4/9
- energy: 9/9

You have nothing in your inventory.

Current goal: Drink Water 

Your output:

{
{
"object": "lava",
"direction": "west",
"distance": 1,
"priority": 0.5
},
{
"object": "water",
"direction": "south",
"distance": 2,
"priority": 1
}
}

Return only the text above and NOTHING ELSE. If no valid output can be generated, return an empty array {}. Your answer should start with '{'

Now consider:

Observation:

''' + obs + "\n\n Current goal:\n\n" + goal + '''

Your Output:
'''
    return LLM_ATTENTION_PROMPT


TRANS_PROMPT = """You are playing an open world survival game. You will be provided with a current goal decribed in natural language. You need to translate that goal into a python condition.

Below is the template of the python condition you should response:

inventory["TODO"] > prev_count

Note that you just need to fill in "TODO".

For example:

Current goal: Drink water
Your output: water

Current goal: Make 2 wood swords
Yout output: wood_sword 

Here are the keys `inventory` contains:

{"health", "food", "drink", "energy", "sapling", "wood", "stone", "coal", "iron", "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"}

If you think you can't use the template to decribe the goal (e.g. navigation task) just reply 'None'.

For example:

Current goal: Locate a stone resource in the environment
Your output: None

Now consider:

"""

EXPLORATION_PROMPT = """
You are playing an open world survival game. You are provided with a current goal. You should generate rules about the game in order to achieve the goal. 

Here are what you can do: 

1. add a new rule;
2. refine a rule;
3. delete an inappropriate rule;

Your response should strictly follow json format. You should use natural language to describe the rules. Please focus on quantitative relationships between objects. Only the rules related to the current goal should be taken into account. 

Here is an example:

{"rule1": "you can take action "place_stone" to place a stone if there is at least one stone in your inventory. "rule2": "building a table requires two wood"}

if you think you can't modify the rule set in current step, please response with the original rule set. you should not generate anything else.

"""
