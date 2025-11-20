Sample agents for the ERC3: AI Agents in action competition.


Make sure to get your ERC3_API_KEY by going to [https://erc.timetoact-group.at/](https://erc.timetoact-group.at/) and entering email with which you've registered ([register here](https://www.timetoact-group.at/events/enterprise-rag-challenge-part-3) and wait 24 hours).



All agents require ERC3 SDK and `ERC3_API_KEY` to connect to the platform and get benchmark. They might also need an `OPENAI_API_KEY` or equivalent.


Here is how you can run sgr-agent-store (simple agent solving `store` benchmark) on your machine:

```
export OPENAI_API_KEY=...
export ERC3_API_KEY=key-.....
cd sgr-agent-store
# activate your virtual environment
pip install -r requirements.txt
python3 main.py
```




