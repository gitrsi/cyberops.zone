![Generative AI for Cybersecurity Professionals Specialization](https://github.com/gitrsi/cyberops.zone/blob/main/assets/img/cyber_generative_ai.jpg "Generative AI for Cybersecurity Professionals Specialization")

> :bulb: Notes on "Generative AI for Cybersecurity Professionals Specialization"



# Generative AI: Introduction and Applications

The creative skills of generative AI come from generative AI models, such as 
- generative adversarial networks or GANs
- variational autoencoders or VAEs
- transformers, and diffusion models

These models can be considered as the building blocks of generative AI. 


## Foundation models

- LLMs
    - OpenAI: GPT n-series
    - Google: PaLM
    - Meta: Llama
- image generation
    - Stable Diffusion
    - DALL-E

## Generative AI tools
- text generation
    - ChatGPT
    - Bard
- image generation
    - DALL-E 2
        - novel images based on textual descriptions
    - Midjourney
    - StyleGAN
        - high quality resolution on novel images
    - DeepArt
        - complex and detailed artwork from a sketch
- audio generation
    - WaveGAN
        - new realistic raw audio waveforms -> speech, music, natural sounds    
    - MuseNet (OpenAI)
        - combine various instruments, styles and genres to create novel musical compositions
        - classical composition to pop songs
    - Tacotron 2 (Google)
        - synthetic speech
    - TTS (Mozilla)
        - synthetic speech
- video generation
    - VideoGPT
        - textual prompt based
        - specify desired content
        - users guide video generation process
    - Synthesia
- code generation
    - Copilot (GitHub)
    - IBM Watson Code Assistant
    - AlphaCode

## History and Evolution of Generative AI

Generative artificial intelligence (AI) is a field that focuses on developing algorithms to create new content, such as text, images, music, and code. Generative AI models are trained on substantial datasets of existing content and learn to generate new content similar to the data they were trained on.

### History
The origin of generative AI origins can be traced back to the initial stages of artificial intelligence exploration. In the 1950s, researchers began to explore the use of computers to generate new data, such as text, images, and music. However, the computational power and data resources needed for these systems to flourish were not yet available.

One of the earliest instances of generative AI dates back to 1964 with the creation of the ELIZA chatbot. Operating on a rule-based system, ELIZA simulated conversations with users by generating responses based on received text. While not genuinely intelligent, ELIZA showcased the potential of generative AI for human-like communication.

During the 1980s and 1990s, hardware and software capabilities advanced considerably and facilitated the development of advanced generative AI models, including neural networks. Neural networks are inspired by the human brain and can learn intricate patterns in data. However, these early neural networks were computationally expensive to train and could only generate small amounts of content.

In the early 2000s, a significant breakthrough occurred in generative AI research with the advent of deep learning. Utilizing neural networks with multiple layers, deep learning models could be trained on extensive datasets to discern complex patterns, enabling the generation of new data that closely resembled human-created content. This breakthrough led to the development of innovative generative AI models, including generative adversarial networks (GANs) and variational autoencoders (VAEs).

GANs and VAEs excel at producing high-quality content that is often indistinguishable from content crafted by humans. 

- GANs operate by training two neural networks in opposition: a generator that creates new content and a discriminator that tries to differentiate between real and synthetic content. Eventually, the generator learns to craft content realistic enough to deceive the discriminator.

- VAEs work by learning a latent space of the data they are trained on. The latent space is a representation of the data that captures the most essential features of the data. VAEs can generate new content by sampling from the latent space and decoding the latent code into the original data space.

In recent years, there has been a rapid explosion in the development of new generative AI models. These models can now generate a wide variety of content, including text, images, music, and code. Generative AI is also used in various applications, such as art, design, and healthcare.

One such instance is the development of diffusion models in 2015. Diffusion models work by gradually adding noise to a clean image until it is completely unrecognizable. They can then be reversed to gradually remove the noise and generate a new image. Diffusion models have been used to create high-quality images and text.

The next significant development is that of large language models (LLMs) like GPT-3and Bardin 2020 and 2023, respectively. LLMs are trained on massive datasets of text and code, which allows them to generate realistic text, translate languages, write different kinds of creative content, and answer your questions in an informative way. 2023 was also when watsonx, a superior generative AI platform based on the cloud, was introduced by IBM. Watsonx can support multiple LLMs.

**1960s: ELIZA**
ELIZA, an early chatbot, showcased early attempts at simulating conversation.

**1980s–1990s: Neural network development**
Researchers started developing more sophisticated generative AI models, including neural networks, capitalizing on advances in hardware and software.

**Early 2000s: Deep learning**
Deep learning, a breakthrough in AI, gained prominence. Neural networks with multiple layers were employed for training on massive datasets.

**2014: Generative adversarial networks (GANs)**
Introduced by Ian Goodfellow and his colleagues, GANs presented a revolutionary two-player neural network framework.

**2015: Diffusion models**
The development of diffusion models brought a novel approach to image generation by gradually adding noise to a clean image.

**2020: GPT-3**
OpenAI released GPT-3, a state-of-the-art language model, demonstrating impressive natural language understanding and generation capabilities.

**2023: Bard and watsonx**
Another large language model, Google’s Bard, and IBM's generative AI system, watsonx are introduced, further advancing the capabilities of generative AI.

These milestones represent an overview of the generative AI journey, capturing vital developments in natural language processing, image generation, and the underlying architectures that have shaped the field over the years. 

### Current scenario

Generative AI is still a relatively young field, but it has already significantly impacted the world. Generative AI is being used to create new forms of art and entertainment, develop new medical treatments, and improve businesses' efficiency. As generative AI advances, its potential societal implications are expected to broaden significantly. 

Presently, these are some of the specific instances highlighting the current applications of generative AI:

- **Art and entertainment:** Generative AI is being used to create new art forms, such as AI-generated paintings, music, and literature. Generative AI is also being used to develop new video games and other interactive experiences.

- **Medicine:** Generative AI is being used to develop new medical treatments, such as personalized cancer therapies and AI-powered drug discovery. Generative AI is also being used to develop new medical imaging tools and improve diagnosis and treatment accuracy.

- **Business:** Generative AI is being used to improve the efficiency of businesses by automating tasks such as customer service, marketing, and sales. Generative AI is also being used to develop new products and services.

Generative AI holds significant transformative potential across various facets of our lives. Using generative AI responsibly and ethically is essential, but it is also important to be excited about its possibilities.


## Capabilities
- text generation
    - Large language modelsLLM
        - trained on large data sets
        - generate human-like text
        - learn patterns and structures from data sets
        - generate coherent and contextually relevant
            - text completion
            - question answering
            - conversation
            - explanations
            - summaries
            - translation
            - image and text pairing
- image generation
    - generative AI images
        - realistic textures
        - natural colors
        - fine grained details
- audio generation
    - musical compositions
    - text to speech
    - synthetic voices
    - natural sounding speech
    - reduce noise, enhance audio quality
    - mimic a human voice
- video generation
    - create basic animations to complex scenes
    - transform images into dynamic videos
    - incorporate temporal coherence
    - exhibit smoot motion and transitions
- code generation
    - complete or create code
        - code snipppets
        - functions
        - complete programs
    - synthesize or refactor code
    - identify and fix bugs
    - test software
    - create documentation
- data generation and augmentation
    - generate new data
    - augment existing data sets
        - images
        - text
        - speech
        - tabular data
        - statistical distribution
        - time series data
    - increase diversity and variability of data
- virtual worlds
    - virtual avatars
        - realistic behavior
        - expressions
        - conversations
        - decisions
    - complex virtual environments
        - realistic textures
        - sounds
        - objects
        - personalized experiences
        - virtual identities with unique personalities

## Applications of Generative AI

### IT and DevOps

Code review
- GitHub Copilot
- Synk DeepCode

Test
- Applitools
- Testim

Monitoring
- AIOps (IBM Watson)
- AIOps (Moogsoft)

CI/CD
- GitLab Duo

other
- natural language interfaces
- automated infrastructure management
- predictive maintenance


### Art and creativity
Generate synthetic content
- music
- scripts
- stories
- videos
- video games

Game development
- Houdini by SideFX
    - games
    - animations
    - AR and VR experiences

Virtual influencers
- interact with users

Education
- content generation
- personalized and adaptive learning
- simulated experiential learning
- language translation
- grade assignments
- create learning journeys
- create assessment strategies
- generate taxonomies
- knowledge tracing
- tutoring support
- virtual and simulated environments
- inclusive education
- NOLEJ
    - AI generated e-learning in minutes
        - interactive video
        - glossary
        - practice questions
        - summary
- Duolingo
    - language learning platform
        - correct french grammar
        - create english test items

### Finance

Banking
- auto detect risks
    - fraud risk
    - credit risk
    - market volatility
- generate insights
- make financially literate recommendations
- KAI-GPT
    - human like financially literate response
- Personetics, AIO Logic
    - detect risks
    - determine rate
    - structure custom loans
    - assessment of customers creditworhiness
    - set credit limits or insurance premiums
BloombergGPT
- analyse
    - news
    - articles
    - social media
- perform market sentiment analysis
- manage investment portfolios

Customer services
- conversational system
- financial planning assistance
    - robo advisors
    - chatbots
    - virtual assistants

Other
- regulatory compliance and reporting
- financial forecasting and prediction
- portfolio optimization
- anti money laundering
- algorithmic trading

### Medicine, healthcare

Medical image analysis
- synthetic images resembling patient data
- synthesize data for rare medical conditions
    
Drug discovery
- generate new molecules
- speed up the process
- lower the development costs

Telemedicine, telemonitoring
- Rasa
    - medically literate conversational system
        - immediate medical advice
        - health related support
        - personalized treatment plans
    
Other
- electronic health record (EHR) management
- healthcare fraud detection
- medical simulation and training

### Human resources

Watson Orchestrate
- automate HR tasks
    - create job requisitions
    - screen and shortlist profiles
    - schedule interviews
    - onboard candidates

Talenteria
- talent acquisition

Leena AI
- HR and employee engagement

Macorva
- workplace and performance management

Other
- training and development
- analytics and decision making
- compliance and policy implementation


## Tools

### Text generation
Based on patterns and structures learned during training, LLMs interpret context grammar and semantics to generate coherent and contextually appropriate text. Drawing statistical relationships between words and phrases allows LLMs to adapt creative writing styles for any given context. 

Multimodal models
- GPT
- PaLM
    combination of a transformer model and Google's Pathways AI platform

Tools
- ChatGPT (GPT based)
    - contextual and relevant responses
    - creativity at work
    - language translation
    - + effective in generating responses and conversational flow 
- Bard (Google PaLM/Path AI based)
    - specialised models for specific tasks
    - sumarize news
    - generate ideas
    - + optimal to research current news of information on a topic

Capabilities
- problem solving through basic mathematics and statistics
- financial analysis, investment research, budgeting
- code generation
 
Other text generators:
- Jasper
    - marketing content tailored to a brand's voice
- Rytr
    - content for blogs, emails, SEO, metadata, ads on social media
- Copy.ai
    - content for social media, marketing and product descriptions
- Writesonic
    - specific templates for different types of text
- Resoomer, Gemini
    - text summarization
- uClassify
    - text classification
- Brand24, Repustate
    - sentiment analysis
- Weaver, Yandex
    - language translation

Data privacy consideration
- AI tools collect and review the data shared with them to improve their systems

Open source privacy-preserving text generators include 
- GPT4ALL
- H20.ai
- PrivateGPT

### Image generation
Generative AI image generation models can generate new images and customize real and generated images to give you the desired output. 

Possibilities
- image to image translation
    - transforming an image from one domain to another
        - sketches to realistic images
        - satellite images to maps
        - security camera images to higher resolution images
        - enhancing detailin medical imaging
- style transfer and fusion
    - extracting style from one image and apply it to another
    - creating hybrid and fusion images
        - converting a painting to a photograph
- inpainting
    - filling in the missing parts of an image
        - art restauration
        - forensics
        - removal of unwanted objects in images
        - blend virtual objects into real world scenes
- outpainting
    - exend an image beyond its borders
        - generating larger images
        - enhancing resolution
        - creating panoramic views

Generative AI models for image generation
- DALL-E by Open AI
    - based on GPT
    - generates high resolution images in multiple styles
    - generates image variations
    - inpainting and outpainting
    - API
- Stable Diffusion
    - open source model
    - high resolution images
    - based on text prompts
    - image to image translation
    - inpainting and outpainting
- StyleGAN
    - enables precise control over manipulating specific features
    - separates image content and image style
    - generates higher resolution images

Free tools
- Craiyon
    - API
- Freepik
- Picsart

Other
- Fotor, Deep Art Effects
    - pretrained styles
    - custom styles
- DeepArt.io
    - photos to artwork
- Midjourney
    - image generator communities
    - exploring each other's creations
    - API
- Bing
    - DALL-E
- Adobe Firefly
    - trained on Adobe stock photos
    - manipulate color, tone
    - lighting composition
    - generative fill
    - text effects
    - generative recolor
    - 3D to image
    - extend image
 
### Audio and video generation
Three categories
- speech generation tools (TTS)
- music creation tools
- tools that enhance audio quality

Tools
- LOVO
- Synthesia
- Murf.ai
- Listnr
- Meta's AudioCraft
- Shutterstock's Amper Music
- AIVA
- Soundful
- Google's Magenta
- GPT-4-powered WavTool
- Descript
- Audio AI
- Runway AI (used in Oscar movie "Everything Everywhere All at Once")
- EaseUS

### Code generation
Generative AI model models and Tools for Code Generation can generate code based on natural language input. Based on deep learning and natural language processing, or NLP, these models comprehend the context and produce contextually appropriate code. 

Tools
- OpenAI's GPT
- Google Bard
- GitHub Copilod (OpenAI Codex)
- PolyCoder (GPT)
- IBM Watson Code Assistang
- Amazon CodeWhisperer
- Tabnine
- Replit

## Glossary
| Term                                     | Definition                                                                                                                                                                                                                                                                     |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Data augmentation                        | A technique commonly used in machine learning and deep learning to increase the diversity and amount of training data.                                                                                                                                                         |
| Deep learning                            | A subset of machine learning that focuses on training computers to perform tasks by learning from data. It uses artificial neural networks.                                                                                                                                    |
| Diffusion model                          | A type of generative model that is popularly used for generating high-quality samples and performing various tasks, including image synthesis. They are trained by gradually adding noise to an image and then learning to remove the noise. This process is called diffusion. |
| Discriminative AI                        | A type of artificial intelligence that distinguishes between different classes of data.                                                                                                                                                                                        |
| Discriminative AI models                 | Models that identify and classify based on patterns they observe in training data. In general, they are used in prediction and classification tasks.                                                                                                                           |
| Foundational models                      | AI models with broad capabilities that can be adapted to create more specialized models or tools for specific use cases.                                                                                                                                                       |
| Generative adversarial network (GAN)     | A type of generative model that includes two neural networks: generator and discriminator. The generator is trained on vast data sets to create samples like text and images. The discriminator tries to distinguish whether the sample is real or fake.                       |
| Generative AI                            | A type of artificial intelligence that can create new content, such as text, images, audio, and video.                                                                                                                                                                         |
| Generative AI models                     | Models that can understand the context of input content to generate new content. In general, they are used for automated content creation and interactive communication.                                                                                                       |
| Generative pre-trained transformer (GPT) | A series of large language models developed by OpenAI. They are designed to understand language by leveraging a combination of two concepts: training and transformers.                                                                                                        |
| Large language models (LLMs)             | A type of deep learning model trained on massive amounts of text data to learn the patterns and structures of language. They can perform language-related tasks, including text generation, translation, summarization, sentiment analysis, and more.                          |
| Machine learning                         | A type of artificial intelligence that focuses on creating algorithms and models that enable computers to learn and make predictions or decisions. It involves designing systems that can learn from training data.                                                            |
| Natural language processing (NLP)        | A branch of artificial intelligence that enables computers to understand, manipulate and generate human language (natural language).                                                                                                                                           |
| Neural networks                          | Computational models inspired by the structure and functioning of the human brain. They are a fundamental component of deep learning and artificial intelligence.                                                                                                              |
| Prompt                                   | Instructions or questions that are given to a generative AI model to generate new content.                                                                                                                                                                                     |
| Training data                            | Data (generally, large datasets that also have examples) used to teach a machine learning model.                                                                                                                                                                               |
| Transformers                             | A deep learning architecture that uses an encoder-decoder mechanism. Transformers can generate coherent and contextually relevant text.                                                                                                                                        |
| Variational autoencoder (VAE)            | A type of generative model that is basically a neural network model designed to learn the efficient representation of input data by encoding it into a smaller space and decoding back to the original space.                                                                  |
















