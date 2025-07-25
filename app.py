import os

import uvicorn
from aidial_client import AsyncDial

from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response, Message, Role

SYSTEM_PROMPT = """You are an essay-focused assistant. Respond to every request by writing a **short essay** of up to 300 tokens.

**Structure:**
- Clear introduction with thesis
- Body paragraphs with supporting points
- Concise conclusion

**Rules:**
- Always write in essay format regardless of topic
- Keep responses analytical and structured
- Use formal, academic tone
- Include specific examples when relevant
- Maintain logical flow between paragraphs
- If user included pictures in request, use them as part of context to generate answers
"""

DIAL_URL = os.environ.get('DIAL_URL', 'http://localhost:8080')
MODEL_NAME = os.environ.get('MODEL_NAME', 'gpt-4o')
API_VERSION = os.environ.get('API_VERSION', '2025-01-01-preview')

class EssayAssistantApplication(ChatCompletion):

    async def chat_completion(
            self, request: Request, response: Response
    ) -> None:
        client: AsyncDial = AsyncDial(
            base_url=DIAL_URL,
            api_key=request.api_key,
            api_version=API_VERSION
        )

        with response.create_single_choice() as choice:
            usr_msg = request.messages[-1].dict(exclude_none=True)
            print(f"User request: {usr_msg}")

            chunks = await client.chat.completions.create(
                deployment_name=MODEL_NAME,
                stream=True,
                messages=[
                    Message(
                        role=Role.SYSTEM,
                        content=SYSTEM_PROMPT,
                    ).dict(exclude_none=True),
                    usr_msg
                ],
            )

            content = ''
            async for chunk in chunks:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        choice.append_content(delta.content)
                        content+=delta.content

        print(f"🤖 Result: {content}\n{'-'*100}\n")


app: DIALApp = DIALApp()
app.add_chat_completion(deployment_name="essay-assistant", impl=EssayAssistantApplication())

if __name__ == "__main__":
    uvicorn.run(
        app,
        port=int(os.environ.get('PORT', '5035'))
    )