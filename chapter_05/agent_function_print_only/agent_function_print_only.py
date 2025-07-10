import httpx
from openai import OpenAI

from readkey import get_token

api_key = str(get_token())
print("API key is : ")
print(api_key)
http_client = httpx.Client(verify=False)
client = OpenAI(base_url="https://api.openai.com/v1", api_key=api_key, http_client=http_client)


def call_open_ai_llm(user_message):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": user_message}],
        temperature=0.7,
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
    )
    print(completion)
    functions=[]
    for choice in completion.choices:
        for completion_message in choice.message.tool_calls:
            functions.append(completion_message.function);
    return functions;


def main():
    try:
        user = "Can you please recommend me a time travel movie?"
        result = call_open_ai_llm(user_message=user)
        print(result)
    except ValueError:
        print("Invalid input. Please enter a number.")


if __name__ == "__main__":
    main()
