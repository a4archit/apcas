from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"

api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version
)


def get_response_from_azure_openai(context: str, query: str) -> str:

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the query from the given context ONLY, if no answer will found then say you don't know."
            },
            {
                "role": "user",
                "content": f"context: {context} \n\nquery: {query}"
            }
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=deployment
    )

    return response.choices[0].message.content




if __name__ == "__main__":

    context = "india have more that 140Cr citizens"
    print(get_response_from_azure_openai(context, query='how many citizens in india'))
    print(get_response_from_azure_openai(context, query='how many states in india'))



