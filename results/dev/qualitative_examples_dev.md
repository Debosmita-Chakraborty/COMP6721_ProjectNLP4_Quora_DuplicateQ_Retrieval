# Qualitative examples (val queries)

Reranking changed the first relevant rank for 307 of 1000 queries: 122 improved, 185 got worse.

## Dense retrieval finds it, BM25 misses it

The labelled duplicate is not in BM25's top 10 but is ranked first by dense retrieval.

*49 queries match.*

**Query:** What are President Obama's plans for his post-presidency?

- Relevant (rank 1 in MPNet): What is President Obama's plan for the immediate future after the elections?
- Top 3 retrieved by MPNet:
  1. What is President Obama's plan for the immediate future after the elections? (relevant)
  2. What will Obama do after his term?
  3. What will Barack Obama do next after he finishes his presidency?

*BM25 first relevant rank: not in top 50; MPNet: 1; word overlap 0.29*

**Query:** What does it feel like to love someone you can't be with?

- Relevant (rank 1 in MPNet): What its like to love someone you can never be with?
- Top 3 retrieved by MPNet:
  1. What its like to love someone you can never be with? (relevant)
  2. How does it feel to be in love with someone you can't never marry?
  3. What is it like to love someone who doesn't love you?

*BM25 first relevant rank: 33; MPNet: 1; word overlap 0.60*

**Query:** Why can't humans ever stay perfectly still?

- Relevant (rank 1 in MPNet): Why can't human keep still?
- Top 3 retrieved by MPNet:
  1. Why can't human keep still? (relevant)
  2. Is it possible to stand still in outer space?
  3. Why do we walk?

*BM25 first relevant rank: not in top 50; MPNet: 1; word overlap 0.40*

**Query:** What do historians do?

- Relevant (rank 1 in MPNet): What does a historian do?
- Top 3 retrieved by MPNet:
  1. What does a historian do? (relevant)
  2. What does an archaeologist do?
  3. What are the differences between an archaeologist and a historian?

*BM25 first relevant rank: not in top 50; MPNet: 1; word overlap 0.33*

**Query:** Why was English never the official language of U.S.?

- Relevant (rank 1 in MPNet): Why has English not been made the national language of the U.S.?
- Top 3 retrieved by MPNet:
  1. Why has English not been made the national language of the U.S.? (relevant)
  2. Why doesn't the USA have an official language?
  3. Why is there no "official" language for the United States?

*BM25 first relevant rank: 43; MPNet: 1; word overlap 0.47*

## The cross-encoder fixes the ranking

The duplicate was not ranked first by MPNet but is ranked first after reranking.

*69 queries match.*

**Query:** What is expected from a product manegement intern at a startup?

- Relevant (rank 1 in MPNet+CE@50): What are product management interns expected to know?
- Top 3 retrieved by MPNet+CE@50:
  1. What are product management interns expected to know? (relevant)
  2. What startups in the San Francisco area are hiring product manager interns?
  3. What is expected of a marketing intern?

*MPNet first relevant rank: 3; MPNet+CE@50: 1; word overlap 0.20*

**Query:** Why are iPhones costly?

- Relevant (rank 1 in MPNet+CE@50): Why iphone are so costly?
- Top 3 retrieved by MPNet+CE@50:
  1. Why iphone are so costly? (relevant)
  2. Why are iPhones so expensive?
  3. Why are iPhones expensive comparing to other mobile brands?

*MPNet first relevant rank: 3; MPNet+CE@50: 1; word overlap 0.50*

**Query:** What Language should I learn first? Spanish, Russian or French?

- Relevant (rank 1 in MPNet+CE@50): Should I learn Spanish, French, or Russian as a second language and why?
- Top 3 retrieved by MPNet+CE@50:
  1. Should I learn Spanish, French, or Russian as a second language and why? (relevant)
  2. Which language should I learn first between French, Italian and Spanish?
  3. Which language should I learn with Spanish: French or Russian? Why?

*MPNet first relevant rank: 2; MPNet+CE@50: 1; word overlap 0.53*

**Query:** Does Hillary Clinton have any specific proposals to help the economy?

- Relevant (rank 1 in MPNet+CE@50): How would Hillary Clinton help the economy as President?
- Top 3 retrieved by MPNet+CE@50:
  1. How would Hillary Clinton help the economy as President? (relevant)
  2. Would the US economy improve with Hillary Clinton's or Donald Trump's campaign proposals?
  3. Does Hillary Clinton have plans to reduce the deficit?

*MPNet first relevant rank: 2; MPNet+CE@50: 1; word overlap 0.33*

**Query:** Which is the best service daikin air conditioner repair center in hyderabad?

- Relevant (rank 1 in MPNet+CE@50): Which is the best service daikin air conditioner service center in hyderabad?
- Top 3 retrieved by MPNet+CE@50:
  1. Which is the best service daikin air conditioner service center in hyderabad? (relevant)
  2. How can we find one of the best Daikin Air conditioner Repair Center in Hyderabad?
  3. WHICH IS THE BEST SERVICE whirlpool air conditioner repair center in hyderabad?

*MPNet first relevant rank: 2; MPNet+CE@50: 1; word overlap 0.92*

## The cross-encoder breaks a correct ranking

The duplicate was ranked first by MPNet but falls lower after reranking.

*125 queries match.*

**Query:** How do I quit my job?

- Relevant (rank 10 in MPNet+CE@50): What is the best way to quit my job?
- Relevant (rank 16 in MPNet+CE@50): What is the best way to quit a job?
- Top 3 retrieved by MPNet+CE@50:
  1. Career Advice: How do I quit the job that I had accepted with lot of hope but it turned out to be some thing else once after I joined the company?
  2. I want to quit my new job. I am not happy at all. But it's been only 2 weeks. What should I do?
  3. I am finally quitting the job I hate! (Yay) anyways I'm new here, how did you guys quit your job? Do you have to write some sort of application or so?

*MPNet first relevant rank: 1; MPNet+CE@50: 10; word overlap 0.20*

**Query:** What are the pros of industrial agriculture?

- Relevant (rank 7 in MPNet+CE@50): What are the advantages of industrial agriculture?
- Top 3 retrieved by MPNet+CE@50:
  1. What are pros and cons of replacing the agriculture sector with the industrial sector in India, as most youth are not willing to opt for agriculture?
  2. What are the pros and cons of agriculture?
  3. What are the pros and cons of industrial urbanization?

*MPNet first relevant rank: 1; MPNet+CE@50: 7; word overlap 0.75*

**Query:** What is it like to work in different parts of the world?

- Relevant (rank 2 in MPNet+CE@50): What is it like to work as an expatriate in different parts of the world?
- Top 3 retrieved by MPNet+CE@50:
  1. What is it like for an Indian to work in different parts of the world?
  2. What is it like to work as an expatriate in different parts of the world? (relevant)
  3. What is it like for an Indian to live and work in other parts of the world?

*MPNet first relevant rank: 1; MPNet+CE@50: 2; word overlap 0.80*

**Query:** What is the best laptop to buy in and around 70000 rupees?

- Relevant (rank 4 in MPNet+CE@50): Which is the best laptop under 70000 Rupees?
- Top 3 retrieved by MPNet+CE@50:
  1. Which is best Laptop to buy in around 60000 Rupees, with best configuration?
  2. Which is the best laptop to buy around 70k INR (rupees)?
  3. Which is the best laptop to buy in a range of 700$, 800$ or around that?

*MPNet first relevant rank: 1; MPNet+CE@50: 4; word overlap 0.43*

**Query:** If light is energy why doesn't it have mass?

- Relevant (rank 4 in MPNet+CE@50): Why does light have no mass?
- Top 3 retrieved by MPNet+CE@50:
  1. If light has zero mass, then as per this [math]E=mc^2[/math], light must have zero energy. Is it so?
  2. If light has no mass, and mass is energy, how does light carry energy? If light has no mass, and momentum is mass multiplied by velocity, how does light carry momentum?
  3. Does Light attract Light? Light is energy therefore E=mc^2. If it has mass it has gravity?

*MPNet first relevant rank: 1; MPNet+CE@50: 4; word overlap 0.33*

