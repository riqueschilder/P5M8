
### Requisitos

Instale as bibliotecas necessarias:
langchain
gradio
python-dotenv
openai
sentence-transformers
chromadb

### Config
Adicione sua chave da OpenAI aqui:

```python
self.llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key="OPENAI_API_KEY", max_tokens=512)

```

### Uso
Rode

```bash
python app.py
```
https://github.com/riqueschilder/P5M8/assets/99187952/cf14707b-c3c6-46b5-9fed-23dc200296bf
