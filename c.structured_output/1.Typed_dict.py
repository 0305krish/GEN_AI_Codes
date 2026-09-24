#import -> model -> schema -> structured_output -> invoke

from typing import TypedDict,Annotated,Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.8-flash"
)

#schema
#data validation could make a issue so shift to pydantic 
class Review(TypedDict):
    summary : Annotated[str, "A brief review"]
    pros : Annotated[Optional[list], "A list of all pros in this"]
    cons : Annotated[Optional[list], "A list of all cons in this"]
    topics : Annotated[list, "A list of all topics talked in this"]
    sentiment : Annotated[str, "Positive, Negative, Neutral"]

structured_model = model.with_structured_output(Review)

response = structured_model.invoke("""I bought the Sony WH-CH720N headphones around three months ago after comparing them with a few other headphones in the same price range. I got the black version for roughly ₹8,500 during a sale, although I have seen the price go up to around ₹10,000 or slightly more depending on the store and discounts. The delivery was quick and the box arrived properly sealed, so there were no complaints with the packaging or initial condition of the product.
The first thing I noticed was the weight. These headphones are surprisingly light, and I can wear them for two or three hours without feeling too much pressure on my head. The ear cups are reasonably comfortable, although after several hours my ears start getting a little warm. The headband is also comfortable but the plastic construction doesn't feel particularly premium. It doesn't look bad, but if you're expecting the kind of solid build you get from more expensive headphones, you might be slightly disappointed.
In terms of sound quality, I would say they are quite enjoyable for everyday listening. The bass is noticeable but doesn't completely overpower the vocals. I mostly listen to pop, Bollywood music, electronic music and occasionally podcasts, and they perform well across all of those. Vocals are reasonably clear and instruments can be distinguished without too much effort. However, people who want extremely detailed or audiophile-level sound may find them lacking. At higher volumes, I also feel the sound becomes a little less clean compared with more expensive headphones.
The noise cancellation is probably one of the main reasons I bought them. On flights, buses and in relatively noisy rooms, the ANC does a good job reducing continuous background sounds such as engine noise or air conditioning. It doesn't magically eliminate every sound, though. Voices and sudden noises can still be heard, so I wouldn't expect complete silence. For studying in a hostel or working in a cafe, I found the noise cancellation useful.
The ambient or transparency mode is also convenient because I don't always have to remove the headphones when someone is talking to me. It lets outside sounds through, although the experience isn't quite as natural as simply not wearing headphones.
Battery life has been another strong point. I usually get several days of use before needing to charge them because I don't listen continuously throughout the day. Even with ANC turned on, the battery lasts long enough that charging has never really been a major concern for me. The charging is through USB-C, which is convenient because I already use a USB-C charger for most of my other devices.
Connectivity has generally been reliable. Pairing with my phone was straightforward, and I didn't experience frequent disconnections. The Bluetooth range is also good enough for moving around my room without keeping my phone in my pocket. The companion app gives some useful controls and lets you adjust the equalizer, which I liked because I could slightly change the sound according to what I was listening to.
The microphone is okay rather than exceptional. For normal phone calls, the other person can understand me, but in a noisy environment the microphone doesn't completely isolate my voice from the background. I wouldn't buy these specifically for professional meetings or gaming communication.
One thing I don't particularly like is the physical construction. The headphones are lightweight, which is great for comfort, but the lightweight plastic also makes them feel less durable than I expected. I haven't broken anything after three months and there are no major scratches, but I am careful when putting them into my bag because I don't want to put unnecessary stress on the hinges.
Another minor issue is that the ear cups can become warm during long sessions. This isn't a huge problem during normal use, but during summer it becomes more noticeable. The controls are fairly easy to learn, although I initially pressed the wrong button a couple of times because there are several buttons close together.
For gaming, they're acceptable for casual gaming but I wouldn't call them a gaming headset. The sound positioning is decent enough for normal games, but competitive players who depend heavily on precise directional audio would probably want something designed specifically for gaming. The lack of a dedicated gaming microphone is also something to consider.
For travelling, however, I really like them. The low weight, ANC and battery life make them convenient to carry around. They also fold somewhat, which makes them easier to put in a backpack. I just wish Sony had included a slightly better carrying case because the included storage solution doesn't feel particularly protective.
When it comes to value for money, I think the price matters a lot. At around ₹8,500 during a sale, I feel the combination of ANC, battery life, comfort and sound quality makes sense. At ₹10,000 or more, I would probably compare them much more carefully with other alternatives before buying. So the value proposition changes depending on the discount.
Pros: lightweight, comfortable for long listening sessions, good battery life, useful ANC, reliable Bluetooth connection, USB-C charging, decent sound quality, useful companion app, adjustable EQ, good for travel and studying.
Cons: plastic build doesn't feel premium, ear cups get warm, microphone is only average, ANC doesn't completely remove voices, buttons can take some time to get used to, carrying protection could be better, and sound quality at very high volume isn't as clean as more expensive headphones.
Overall, after using them for about three months, I would describe them as a practical everyday pair of headphones rather than a premium or audiophile product. They are particularly useful for studying, travelling, watching videos and listening to music for long periods. The biggest strengths for me are the comfort, ANC and battery life, while the biggest weaknesses are the plastic build and average microphone. I haven't had any major technical problems so far, and if someone finds them at a good discount, they could be worth considering, but I wouldn't say they are perfect for every type of user.""")
print(response)
