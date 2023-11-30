import re

import telebot
import time
import uuid
import requests
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain import FAISS
from faiss import IndexHNSWFlat
from typing import Tuple

from telebot.formatting import escape_markdown

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

API_TOKEN = ''
bot = telebot.TeleBot(API_TOKEN)
openai_api_key = ""
gigachat_api_key = ""
data_path = 'data/langchain_dataset.csv'


class GigaChatSecureToken:
    access_token: str
    expires_at: int
    _offset: int = 60  # If token will be expired in {offset} seconds, is_expired() return true

    def __init__(self, access_token: str, expires_at: int):
        self.access_token = access_token
        self.expires_at = expires_at

    def is_expired(self):
        return round(time.time() * 1000) > self.expires_at + self._offset * 1000

class GigaChatLLM(LLM):
    api_key: str = None
    temperature: float = 0.7
    secure_token: GigaChatSecureToken = None

    @property
    def _llm_type(self) -> str:
        return "gigachat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if not self.secure_token or self.secure_token.is_expired():
            print("Obtaining new secure token")
            self._auth()
            if not self.secure_token or self.secure_token.is_expired():
                # New token was not obtained
                print("ERROR: new token was not updated, cannot call LLM")
                return ""
        headers = {
            "Authorization": f"Bearer {self.secure_token.access_token}",
            "Content-Type": "application/json"
        }
        req = {
            "model": "GigaChat:latest",
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "temperature": self.temperature
        }
        response = requests.post("https://gigachat.devices.sberbank.ru/api/v1/chat/completions", headers=headers, json=req, verify=False)
        if response.status_code != 200:
            print(f"ERROR: LLM call failed, status code: {response.status_code}")
            return ""
        return response.json()["choices"][0]["message"]["content"]

    def _auth(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded"
        }

        scope_info = {"scope": "GIGACHAT_API_PERS"}
        response = requests.post("https://ngw.devices.sberbank.ru:9443/api/v2/oauth", data=scope_info, headers=headers, verify=False)
        if response.status_code != 200:
            print(f"ERROR: Something went wrong while obtaining secure token, status code: {response.status_code}")
            return
        content = response.json()

        expires_at = content["expires_at"]
        token = content["access_token"]
        if not (expires_at and token):
            print("ERROR: server returns empty values for fields 'expires_at' or 'access_token'")
            return
        self.secure_token = GigaChatSecureToken(token, expires_at)

llm = GigaChatLLM(api_key=gigachat_api_key)

embedding_type = 'openai'
token_counts_max = 360
openai_batch_size = int(150000 / (3 * token_counts_max))
embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key, chunk_size = openai_batch_size)

index_name = f'{data_path}_{embedding_type}'
index = IndexHNSWFlat()
faiss = FAISS(embeddings, index, None, None)

try:
    db = FAISS.load_local(index_name, embeddings)
except:
    raise Exception(f'Индекс {index_name} не найден')

retriever = db.as_retriever(search_kwargs = {"k": 5, "score_threshold": 0.3})
prefix = """
Ответь на вопрос: {query}

При ответе учитывай следующие имеющиеся данные:
"""

context_example_template = """
{context}
"""

suffix = "Если вопрос тесно связан с предоставленными данными, используй их при формировании своего ответа."

def escape_f_string(text):
  return text.replace('{', '{{').replace('}', '}}')
    
def escape_examples(examples):
    return [{k: escape_f_string(v) for k, v in example.items()} for example in examples]
    
def get_answer(question: str) -> Tuple[str, str]:
  res = retriever.get_relevant_documents(question)
  examples = [{"context": doc.page_content } for doc in res ]
  context_example_template_prompt = PromptTemplate(
      input_variables=["context"],
      template=context_example_template
  )

  example_selector = LengthBasedExampleSelector(
    examples=escape_examples(examples),
    example_prompt=context_example_template_prompt,
    max_length=900
  )

  few_shot_prompt_template_prompt = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt=context_example_template_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
  )

  chain = few_shot_prompt_template_prompt | llm

  answer = chain.invoke({ "query": question })


  prompt = few_shot_prompt_template_prompt.format(query = question)
  return answer, prompt
    

@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(
        message.chat.id,
        text=f'Привет, {message.from_user.username}!\n'
             f'Это контекстная LLM модель работающая на базе GigaChat API. '
             f'Я могу отвечать на любые вопросы, но лучше всего у меня '
             f'получается отвечать на вопросы по работе с фреймворком LangChain\n\n'
             f'Спроси меня что-нибудь, например "Как сохранить настроенные цепочки как собственные объекты в LangChain?"'
    )

# Переменная для хранения времени последнего вызова
last_call_time = None

# Время "тишины" между вызовами в секундах
cooldown = 20


def format_code_if_needed(text: str) -> str:
    formatted_text = re.sub(r"<code\\>\\{\w+\\}", "```\n", text)
    formatted_text = re.sub(r"</code\\>", "```", formatted_text)
    return formatted_text


@bot.message_handler(content_types=['text'])
def handle_question(message):
    global last_call_time
    current_time = time.time()
    # Проверяем, пройдено ли 20 секунд с последнего вызова
    if last_call_time is not None and current_time - last_call_time < cooldown:
        bot.reply_to(message, "Бот занят, попробуйте через 20 секунд")
        return
        
    last_call_time = current_time
    user_question = message.text

    bot.send_message(message.chat.id, "Вопрос принят, скоро придет ответ, пожалуйста, подождите :)")
    answer, prompt = get_answer(user_question)
    formatted_answer = format_code_if_needed(escape_markdown(answer))
    bot.send_message(message.chat.id, formatted_answer, "MarkdownV2")
    bot.send_message(message.chat.id, "Надеюсь мой ответ был полезен. Я готов ответить на следующий вопрос.")


bot.infinity_polling(timeout=10, long_polling_timeout = 5)
