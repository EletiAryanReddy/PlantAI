import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def ask_bot(question):

    completion = client.chat.completions.create(

        model="llama3-8b-8192",

        messages=[
            {
                "role": "system",
                "content": "You are an agriculture AI assistant."
            },
            {
                "role": "user",
                "content": question
            }
        ]

    )

    return completion.choices[0].message.content