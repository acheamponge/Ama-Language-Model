import streamlit as st
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from PIL import Image


from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")



data = """
They had asked me, 
a worthy friend and
a loving brother,
to “stop shouldering
the world’s troubles,”\n
one meaning Africa,
the other women
“learn to
laugh and
live!”\n
I grow hot:
thinking that
laughing?\n
That’s easy:
it’s all we do instead of
crying.\n
And since there’s
so much to cry about
we laugh and
laugh and
laugh.\n
But
living?
You could tell them
that’s not easy.\n
In a real life
in a real world
perhaps.\n
But here
where
on a bare belly
for less than a cedi,\n
you gathered
in single pieces and
carried\n
ten bushels of
solid stones
your four-month-old baby
straddled on
your back,\n
slipped,
fell
broke your
arm—? \n
Laughing we do for
fear of
crying.\n
Living
we don’t discuss
here.\n
My uncle was the prophetic one,
throwing his beads this way and that,
diving, foretelling,
warnings galore, sweet promising.\n
One eye on the past, four to the future,
half a dozen or more for now.\n
He was good if the news was good;
for evil news we blamed the beads.\n
Made from bones
or fashioned glass,
cut out from stones
or beaten brass\n
It’s the many human hours, Sister,
it’s the sweat and blood, Brother,
which makes the bead a thing apart
from precious diamonds, opals, and gold.
Turn them this way, shake them that way,
see how they shine incandescent,
see how they glow
in a million hues.\n
Elegant and enchanting bead,
flowered flawed, folded, or fielded,
you are the true frame of our feasts,
your festivals, fetes, and fiestas.
Give me a bead that’s wrapped in joy;
find me a bead to carry my grief.\n
We sing of beads, and sing with beads;
just see how well they show on us.
Beads are the zeze of our joyous trails,
the ziz of life when all else fails.
Beads are zany, zesty, zingy,
the greatest zaiku, a grief zapper.\n
Speak to me of beads, Grandma,
speak to me.\n
Talk to me of beads, Nana,
talk to me.\n
She brightened up immediately,
she looked at me with a welcome smile.\n
Grandma pulled up a stool and sat,
she listened well to me and asked:
“You want a tale on beads, do you?\n
You want a tale or two?
I’ll tell a tale or two to you.\n
But to speak to you of every bead,
in words that sing and dance like them,\n
you and I shall surely need
more than my life in hours and days,
more than your life in weeks and years.\n
A million lifetimes is not much
if beads are the theme, the thought, the thing.\n
We dive for beads, we swim, we float,
we mine for beads, we comb the woods.\n
Koli beads for the infant
on his wrist and on her waist,\n
cascades of white beads for the mother,
a very fitting celebrant.\n
There are beads that are tame
like what welcomed baby here;\n
there are beads that are wild,
lion’s teeth, lightning struck.\n
And there are beads around my waist,
For only my and my dot-dot’s eyes!!\n
Have you seen my love tonight?
Asked the ardent warrior youth.\n
Light of step, curved like a bow,
her eyes were wonders to behold.\n
She was oiled and very clean,
she was powdered like a queen,\n
she wore a sarong of the purest silk,
her toes were nestled in their thongs.\n
Have you seen my love tonight?
She who wore gold beads in her hair?\n
Then the pretty maiden asked,
who has seen my love tonight?\n
Who has seen my warrior brave?
he had said no more to war,
he had buried his arrowhead.\n
His girdle was free of blood and sweat.\n
He was adorned in his very best,
he was oiled like a king,
with beads of silver in his hair.\n
Who has seen my love tonight?
They welcome us here in the palest white
and bid us farewell in black,\n
sometimes blue, and brown, and red,
metallic green, or indigo.\n
There are beads, by far the most,
that are polished, tarred, and feathered.\n
There are beads, worked over and under,
elegant hued, thin and narrow.\n
Beads are the zaffered, the zingiest,
the zenith of all great times.\n
Cool, calm, and forever collected,
clawed, clayed, or colored,\n
constantly changing, bead
you are the best, you are the greatest.\n
So don’t talk to me of the chevron.\n
Don’t ever talk of it.\n
Don’t break my ears on the chevron.\n
Don’t break my ears!!!\n
As barter for my life and yours,
no gem on earth could fit the bill.\n
Not gold, and if not even gold,
then what on earth is chevron?\n
I dread the chevron.\n
It was a weapon
of oppression,
and not at all . . . a bead.\n
Seven whole humans for one bead?\n
And what kind of trade was that?\n
A layer each of sand and mud\n
for the lives of our kinsmen?\n
So what if it was one and not seven?\n
One soul for a shiny piece of bead?\n
This sounds like the greatest greed,
this sounds like utter foolishness!\n
Don’t talk to me of the chevron,
don’t even mention it.\n
Don’t break my ears on the chevron,
don’t break my ears.\n
They say that cheap beads prattle,
rattle, and tattle,
but great beads never talk.\n
Yet if a string of beads is fine,
it sings,
it dances,
it jumps,
and sizzles.\n
If a string of beads is truly fine,
it can speak in a million tongues.\n
It will have something for all,
and say the most amazing things.\n
And every now and every then
every bead laughs out aloud.\n
There are beads that are smaller
than the hopes of a mean mind.\n
Though called bodom, as in a dog,
poochy pug, puggy pooch,
bodom beads, they are so big,\n
they are the elephants of the pack.\n
They lead the way
and announce the day.\n
The nature of beads is a mystery,
the how of it, the feel, the glow
of earthly gems: the least and most,
our first and true try to create, to beautify our human selves.\n
The best of doors to human hearts,
our spirit’s window to the world,
beads clothe our woes in vivid color.\n
Beads like angels plead for us.\n
Beads can lift the heaviest heart.\n
And like tea and precious brews,
beads can warm us when we are cold,
and cool us when we are hot.\n
Blessed are the beads
that bring us peace.\n
Spare us, O Lord, in this lifetime,
beads of war, chaos, and strife.\n
No beaded strings of calamities,
earthquakes, floods, and famine.\n
No veritable tsunamis of woe.\n
Keep us cool and keep us warm.\n
For each color in the rainbow,\n
there is a bead, somewhere on earth:
a million years old, if a day,
or shy in its newsness, and done this dawn.\n
Blue beads, green beads,
yellow beads and grey,
black beads, white beads,
red beads and brown.\n
Your rise from heaps of your own ash
with more of you than ever were.\n
You, bead, are an awesome one,
you are the phoenix of the years.\n
Their making uses endless hours,
the how, the when, the what of it.\n
The wearing is by a billion souls
whichever way, however much, and everywhere . . .\n
Mined and molten
man-made wonder,
raw organic, or cooked, and dried,
forever treasured, forever prized.\n
Bettered and bartered,
broken and beaten,
burnt or badgered,
bruised and bloodied
you are the never-left-behind,
oldest, ordered, owned invention.\n
Pure and precious, polished pearl,
still safe, sacred, scraped, or scratched;\n
Traded, treated, tough in trouble,
unique, unmatched, unbreakable.\n
Verdant velvet, virginal as rain,
beads are virile, vestal, vain.\n
Gilded and golden,
there can be no palanquin.\n
If you are not sitting with the king,
you are the queen,
the soul, and spirit within.\n
Beads are deserving,
beads are worthy,
wash me some beads to warm my skin,
a token of love, a gift for my kin.\n
Hollowed and hallowed,
jingled, jangled, juggled,
you are our life’s companion,
the closest friend until the end.\n
Don’t tell me if there were no beads
something else could meet our needs.\n
Something what? Something where?
Please keep it there, even if it’s rare.\n
Who was it said
the reason why
you never see
Black Folks properly
e-v-e-r on film or TV
is ’cause White Folks
“find them threatening”?\n
Whopei! Abae-o-o-o!
We always thought
our beautiful black skin
was
the Problem.\n
so
Afia and Ola
Eye-leen, Lola, and Tapu
bleached and blotched
their skins ugly
to please our masters and our masters’ servants.\n
Now
don’t come telling me
flat noses,
thick lips, and
small ears
must also disappear
to put the world at ease?\n
That must explain
why the Princess Nefertiti
and the youthful King Tut
were dragged to
Michael Jackson’s beauty doctor
long before
Young Michael was born,\n
and also why
the Sphinx
who looked like
Great Ancestor King Khafre
is being redone!\n
We should have known
we were in trouble
the day we heard
a Corsican general traveled to Giza
by way of Paris and a crown
to shoot
the Sphinx’s nose off
for not-at-all-looking like
his.\n
Enfin! Helas!! Mon Dieu!!!
Ebusuafo,
for years
the Sphinx stood
massive eternal
riddled with wisdom and all
very thick-lipped
very flat-nosed.\n
We never saw him photographed head-on.\n
But in the year 2020
the New Sphinx will be unveiled
full visage on view
straight nose raised
thin lips tight
and even, maybe, blue-eyed:\n
a perfect image of the men
who vested so much interest
in his changing face.\n
You see, Wekumei,
when folks figure
you are their slave
your past belongs to them.\n
And mind you, the Man will try
to grab our future too.\n
Shall we let him?\n
Come near, come round,
My dear sisters
My dear brothers.\n
Come take a peek, and
Have a glance\n
Someone is claiming to be dead
And she looks like Juliana! \n
Here we are, straw on straw,
clutching at each one in the dark,\n
wondering if it’s true, and if it’s a fact
which is the ugliest in this news.\n
If what has been is what now is,
Then darkness is here and may never leave.\n
The pot is broken for the clan
The house is shaken, its fires long dead.\n
Hear the screaming tornado,
The boastful roaring hurricane:\n
at the workplace
in the market,
here by the hearth at home.\n
Esi Boah Bediako Dwemoh
Our beautiful singular Juliana
It’s true we shall all die one day,\n
Did you have to leave this soon?             
Dear All-consuming Mother Earth,
Voracious, indiscriminate,\n
Carnivorous cannibal,
We wish this meat you would spit out,
Offer us back our Juliana:\n
Beautiful
Winsome
Joyous.\n
Kind and thoughtful Mrs. Dwemoh\n
Please listen, and listen good:
If you dreamt\n
You could skip town
Sneak out of our lives
And disappear,\n
Happy you’d finished
Your allotted tasks,
Then Mighty Mama, think again.\n
For us your task if not yet done.\n
Trainer of trainers
The teachers’ teacher\n
We know why
This noon in the tropics\n
Schoolrooms are silent
The children abandoned.\n
The wisest counselor
Accomplished crafter
Sower of hope and joy and peace
You cloaked us in love and habiliments.\n
Good Mother Juliana
Are you sure you want to leave?
Come near, come round,
My dear sisters
My dear brothers.\n
Come take a peek, and
Have a glance
Someone is claiming to be dead
And she looks like Juliana!\n
Aunt.
Don’t ask
me how
I come to address my mother thus.\n
Long
complex, complicated stories:\n
heart-warmingly familial and
sadly colonial.\n
You know how
utterly, wonderfully\n
insensitive the young can be?
Oh no. We are not here talking adults
who should know better
but never do.\n
Aunt,
I thank you for
being alive today, alert, crisp.\n
Since we don’t know tomorrow,
see me touching wood,\n
clutching at timbers, hugging forests:
So I can enter young,
age, infirmities
defied.\n
Hear my offspring chirping:
“Mummy, touch plastic,
it lasts longer!”\n
O, she knows her mama well.\n
The queen of plastics a tropical Bedouin,
she must travel light.\n
Check out the wood,
feel its weight, its warmth
check out the beauty of its lines, and perfumed shavings.\n
Back to you, My Dear Mother,
I can hear the hailing chorus
at the drop of your name.\n
And don’t I love to drop it
here, there, and everywhere?\n
Not missing out by time of day,
not only when some chance provides,
but pulled and dragged into talks
private and public.\n
Listen to the “is-your-mother-still-alive” greeting,
eyes popping out,\n
mouth agape and trembling:
That here,
in narrow spaces and
not-much-time,\n
who was I to live?\n
Then she who bore me?
Me da ase.
Ye de ase.\n
"""

def generate_seq(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    return in_text

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
# create line-based sequences
sequences = list()
for line in data.split('\n'):
    encoded = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)
# pad input sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
# split into input and output elements
sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)


st.title('Ama Ata Aidoo Machine Learning Project')

st.write(
        f'<iframe width="560" height="315" src="https://www.youtube.com/embed/l6ittGqSZ9k" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
        unsafe_allow_html=True,
    )
st.write('This project is a Text Generator of words trained of the poetry of the Literary Genius Ama Ata Aidoo')

n = st.number_input('Type the number of words you want generate', min_value=1, step=1 )


s = st.text_input('Type a word or words you want to generate after')

if s and n:
    st.header((generate_seq(loaded_model, tokenizer, max_length-1, s, n)))

elif s and not n:
    st.write('Please input information')

else:
    st.write('Please input a word and a number')
    
    
