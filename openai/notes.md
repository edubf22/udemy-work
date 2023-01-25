# Playground

No code required, just need to login. Initially free, but after 3 months or a number of tokens (whatever comes first) user must pay to send other requests.

choose mode: completion, insert, and edit

model: davinci, ada, and others. Pricing is different depending on model

View code: there is an option at the top that allows you to check the code in language of choice, such as python, node.js, curl, or check json file

"Best of" function: generates multiple answers (be careful when getting billed)

# Prompt Engineering

## Simple questions

* Provide sentence with a question mark, or fullstop, so the model knows how to respond (not to complete)
Ex: What color are fire engines in the UK?

## Generate some ideas

* Use colon after a statement
Ex: Brainstorm some ideas combining VR and fitness:

## Simple completion
* Leave sentence unfinished and model will continue from where we stopped. Be carefull by number of tokens generated
Ex: Vertical farming provides a novel solution for

## Classification
Provide a sentence and ask GPT to classify it.
Ex: Tweet: I loved the new Batman movie!
Sentiment:

## Multiple Requests
Provide a list of items.
Ex: Classify the sentiment in these tweets:
1. "I can't stand homework"
2. "This sucks, I'm bored"
3. "I can't wait for Halloween!!!"

Tweet sentiment ratings:
1. 

## Grammar correction
Ex: Correct this to standard english:
She not want to sleep.

## Summarization
Ex: Summarize this text for a child:

bla bla bla 

## Translation
Language translation, Codex (Code translation)

# Improving GPT prompts to get better responses
Be explicit
 - Show and tell:
 Write a poem about OpeanAI vs. Write a short inspiring poeam about OpenAI, focusing on the recent DALL-E product launch (DALL-E is a text to image ML model) in the style of Shakespeare
 
 - Give more example responses:
 Extract the important entities mentioned in the text below. First extract all copmany names, then extract all people names, then extract specific topics which fit the content and finally extract general overarching themes.
 
 Desired format:
 Company names: -,-
 People names: -,-
 Specific topics: -,-
 General themes: -,-
 
 Text: {text}
 
 - Reduce fluff (be precise)
 
 - Positive instruction:
 Tell the AI what to do instead of what NOT to do
 
 - Lead the way:
 For example, in Codex, provide some lines with specific programming language.
 
 - Adjust tone by adding tags such as polite response
  
 - Provide quality data
  
 - Use "temperature" and "top_p" settings":
 Temperature is high, the AI can choose freely the words (helpful for creative writing), avoid conversation drift
 Top_p to avoid long tail
 
 - Position of \n matters (completion or generated a new answer, such as in a dialogue)
 
 - Create smaller tasks (create outline, write intro paragraphs, write a paragraph about..., etc) vs. one large task (write an article about...)

# Prompt Design - Templating
Enumerate tasks to help the model. Example:
"""
1. You are an assistant helping...
2. ...to generate a <insert task here, e.g. write a blog post>...
3. ...based on a prompt (e.g. use this format, replacing text in brackets with the result. Do not include the brackets in the output.)

4. Blog post:
5. #[Title of blog post]
[Content based on the prompt]
6. Prompt itself followed by a new line.
"""

Example:
"""
You are an assistant helping to generate a blog post based on a prompt. Use this format, replacing text in brackets with the result. Do not include the brackets in the output:

Blog post:
# [Title of blog post]
[Content of blog post]
"""
Blue Snowball\n

## Summarizing text
Summarize this for a 3rd grader. 
"""
Provide instructions inside the comment
"""

# Code examples 
## Simple CURL example
```
curl https://api.openai.com/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API" \
  -d '{
  "model": "text-davinci-003",
  "prompt": "Say this is a test",
  "temperature": 0.7,
  "max_tokens": 256,
  "top_p": 1,
  "frequency_penalty": 0,
  "presence_penalty": 0
}'
```

## Simple Python Example
- Install the library (`pip install openai`)
```py
import os
import openai

openai.api_key = os.getenv('OPENAI_API')

response = openai.Completion.create(model='text-davinci-003', prompt='Say this is a test', temperature=0, max_tokens=7)
```

## Simple Node.js example
- Install the library (`npm install openai`)
```
const { Configuration, OpenAIApi } = require("openai");

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

const response = await openai.createCompletion({
  model: "text-davinci-003",
  prompt: "Say this is a test",
  temperature: 0,
  max_tokens: 7,
});
```

## Simple C# example
- Install the library (`Install-Package OpenAI`)
```
var api = new OpenAI_API.OpenAIAPI("sk-myapikeyhere", Engine.Davinci);

var result = await api.Completions.CreateCompletionAsync("Say this is a test", temperature: 0);

// should print something like "\n\n This is indeed a test"
Cnosole.WriteLine(result.ToString());
```

## Powershell example
```
$apiEndpoint = "https://api.openai.com/v1/completions"
#apiKey = "sk-APIKEY" 

$headers = @{
  Authorization = "Bearer $apiKey"
}

$contenttype = "application/json"

$body = @{
  model = "text-davinci-003"
  prompt = "This is only a test"
} | ConvertTo-Json

$response = Invoke-WebRequest -Uri $apiEndpoint -Method POST -Headers $headers -Body $body -ContentType $contenttype

$generatedText = $response.Content | ConvertFrom-Json

Write-Output $generatedText.choices[0].text
```

# The Moderation Endpoint
This is the only free-to-use endpoint and can be used for completions.
`POST https://api.openai.com/v1/moderations/`

Need to provide some text ('input'). The moderations endpoint looks at how harmful the input could be to the user. There are different categories, such as hate, self-harm, sexual, etc. 

Check out the `moderation.ipynb` notebook for an example. 

# Completion parameters
## How tokens work
Each token is given a probability score by the AI, and response is created by joining tokens together that have high probabilities (this can be adjusted with code).

## Calling the API

```
response = openai.Completion.create(
 model="text-davinci-003",
 prompt="What is human life expectancy in the US?",
 max_tokens=500,
 temperature=0.1
)
```
We can adjust the request by changing parameters such as `max_tokens` or `temperature`.
* `max_tokens`: Only cuts the output once the max value is reached, also counts input/query tokens
* `top_p`: default is 1, 0.9 allows AI to be more creative, 0.3 consider only the top 30% by mass.
* `temperature`: 0 for well defined answers (token with highest probability), 0.9 for creative applications, 2 would give all tokens same probability 
* `n`: generate multiple answers (completions), default is 1 - be careful as multiple responses will consume many tokens
* `stop`: preserves tokens by avoiding verbose responses, defaults to null. Examples: stop=['\n', 'Human:', 'AI:']
* `best_of`: generates multiple answers on server end, returns the best one (default is 1). This can be combine with `n` to send more than 1 response. This also counts towards the tokens. 
* `suffix`: Tell the AI what will follow, limited to 40 characters. 
* `echo`: include the prompt in the completion (default is false, so doesn't include text that was used in the prompt).
* `user`: optional, useful for policy violations.

# Adjusting probabilities 
Each token represents a word, or part of a word. There are about to 50,000 tokens in GPT-3. We can check the tokens ![here](https://beta.openai.com/tokenizer).

## logprobs
We can ask the model what are the highest probability tokens within the response. The tokens are given with a log of the probability (logprob), so the close to 0, the higher the probability.

## Banning tokens - logit_bias
Assign a Bias by deducting 100 (-100) from a token's logprob. Example:
```
response = openai.Completion.create(
 model='text-davinci-003',
 prompt='What is the capital of France?',
 logit_bias: {'6342': -100}
)
```
This would change the logprob for token_id = 6342. We could also address this by adding (instead of deducting) a value to the logprob of the token of interest. 

## Dealing with repetition

