import httpx
from openai import OpenAI
import json
from readkey import get_token

api_key = str(get_token())
print("API key is : ")
print(api_key)
http_client = httpx.Client(verify=False)
client = OpenAI(base_url="https://api.openai.com/v1", api_key=api_key, http_client=http_client)

# Function to be called for each LLM request
def recommend(topic, rating="good"):
    if "time travel" in topic.lower():     #1
        return json.dumps({"topic": "time travel",
                           "recommendation": "Back to the Future",
                           "rating": rating})
    elif "recipe" in topic.lower():    #1
        return json.dumps({"topic": "recipe",
                           "recommendation": "The best thing … ate.",
                           "rating": rating})
    elif "gift" in topic.lower():      #1
        return json.dumps({"topic": "gift",
                           "recommendation": "A glorious new...",
                           "rating": rating})
    else:     #2
        return json.dumps({"topic": topic,
                           "recommendation": "unknown"})

def call_open_ai_llm_with_message(messages):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    print(completion)
    return completion.choices[0].message.content;
def call_open_ai_llm_with_tools(messages,tools):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=messages,
        # messages=[
        #     # {"role": "system", "content": "You are a helpful assistant."}, >>> This is the difference , we are removing the role system to execute #recommend function
        #           {"role": "user", "content": user_message}],
        temperature=0.7,
        tool_choice="auto",
        tools=tools
    )
    return completion;


def list_available_functions():
    available_functions = {
        "recommend": recommend,
    }  # only one function in this example, but you can have multiple
    return available_functions;

def get_function_tool_llm_data():
    tools=[{
            "type": "function",
            "function": {
                "name": "recommend",
                "description": "Provide a … topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description":
                                "The topic,… for.",  # 5
                        },
                        "rating": {
                            "type": "string",
                            "description":
                                "The rating … given.",  # 5
                            "enum": ["good",
                                     "bad",
                                     "terrible"]  # 6
                        }
                    },
                    "required": ["topic"],

                }
            }
        }]
    return tools;

def main():
    try:
        user = """Can you please make recommendations for the following:1. Time travel movies2. Recipes3. Gifts"""
        messages = [{"role": "user", "content": user}]

        # #Get the function names and matching only, not calling actual call
        function_call_only_response = call_open_ai_llm_with_tools(messages=messages,tools=get_function_tool_llm_data())
        print(function_call_only_response)
        response_message = function_call_only_response.choices[0].message

        # Iterate through each matched function
        tool_calls = response_message.tool_calls
        # Step 2: check if the model wanted to call a function
        if tool_calls:
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            messages.append(response_message)
            # Step 4: send the info for each function call and function response to the model
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = list_available_functions()[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    topic=function_args.get("topic"),
                    rating=function_args.get("rating"),
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response

            second_response = call_open_ai_llm_with_message(messages=messages)
            print(second_response)
    except ValueError:
        print("Invalid input. Please enter a number.")


if __name__ == "__main__":
    main()
