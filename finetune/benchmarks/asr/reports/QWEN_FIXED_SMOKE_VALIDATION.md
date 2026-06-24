# Qwen Fixed Smoke Validation

Run: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_185009_qwen3_asr_1_7b_fixed_text_extraction_smoke_smoke`

Rows: 64
Malformed hypotheses: 0
Duplicate IDs: 0
Extraction paths: `['$[0].text']`
Result types: `['qwen_asr.inference.qwen3_asr.ASRTranscription']`

## Metrics

Normalized WER: 0.037165082108902334
Raw WER: 0.9818496110630942
Normalized CER: 0.013495934959349594
SER: 0.3125
Exact match: 0.6875

## Sampled Pairs

- `7127-75947-0007` `test-clean` WER `0.0`
  - ref: SHE THEN ROSE HUMMING THE AIR TO WHICH SHE WAS PRESENTLY GOING TO DANCE
  - hyp: She then rose, humming the air to which she was presently going to dance.
- `5683-32879-0009` `test-clean` WER `0.0`
  - ref: ILL AND TROUBLED DEAR TROUBLED IN MIND AND MISERABLY NERVOUS
  - hyp: Ill and troubled, dear, troubled in mind and miserably nervous.
- `2300-131720-0023` `test-clean` WER `0.02631578947368421`
  - ref: I THINK HE WAS PERHAPS MORE APPRECIATIVE THAN I WAS OF THE DISCIPLINE OF THE EDISON CONSTRUCTION DEPARTMENT AND THOUGHT IT WOULD BE WELL FOR US TO WAIT UNTIL THE MORNING OF THE FOURTH BEFORE WE STARTED UP
  - hyp: I think he was perhaps more appreciative that I was of the discipline of the Edison Construction Department, and thought it would be well for us to wait until the morning of the fourth before we started up.
- `1221-135767-0021` `test-clean` WER `0.0`
  - ref: PEARL ACCORDINGLY RAN TO THE BOW WINDOW AT THE FURTHER END OF THE HALL AND LOOKED ALONG THE VISTA OF A GARDEN WALK CARPETED WITH CLOSELY SHAVEN GRASS AND BORDERED WITH SOME RUDE AND IMMATURE ATTEMPT AT SHRUBBERY
  - hyp: Pearl accordingly ran to the bow window at the further end of the hall, and looked along the vista of a garden walk carpeted with closely shaven grass, and bordered with some rude and immature attempt at shrubbery.
- `1320-122617-0012` `test-clean` WER `0.0`
  - ref: FOUR OR FIVE OF THE LATTER ONLY LINGERED ABOUT THE DOOR OF THE PRISON OF UNCAS WARY BUT CLOSE OBSERVERS OF THE MANNER OF THEIR CAPTIVE
  - hyp: Four or five of the latter only lingered about the door of the prison of Uncas, wary but close observers of the manner of their captive.
- `8461-281231-0026` `test-other` WER `0.0`
  - ref: HERE IS A BUGLE WHICH AN ENGLISH YEOMAN HAS ONCE WORN I PRAY YOU TO KEEP IT AS A MEMORIAL OF YOUR GALLANT BEARING
  - hyp: Here is a bugle which an English yeoman has once worn. I pray you to keep it as a memorial of your gallant bearing.
- `6070-86745-0017` `test-other` WER `0.0`
  - ref: THEY SAY THAT IT IS QUITE FAIR AND THAT SOWING SO MUCH RED YOU OUGHT TO REAP A LITTLE BLUE
  - hyp: They say that it is quite fair, and that sowing so much red, you ought to reap a little blue.
- `6128-63240-0010` `test-other` WER `0.058823529411764705`
  - ref: WELL SO IT IS THEY ARE ALL WITCHES AND WIZARDS MEDIUMS AND SPIRIT RAPPERS AND ROARING RADICALS
  - hyp: Well, so it is. They are all witches and wizards, mediums and spirit wrappers, and roaring radicals.
- `7018-75788-0014` `test-other` WER `0.0`
  - ref: HERE I FOUND A GREAT SHIP READY FOR SEA AND FULL OF MERCHANTS AND NOTABLES WHO HAD WITH THEM GOODS OF PRICE SO I EMBARKED MY BALES THEREIN
  - hyp: Here I found a great ship ready for sea, and full of merchants and notables, who had with them goods of price. So I embarked my bales therein.
- `6432-63723-0010` `test-other` WER `0.0`
  - ref: AARON GRAFTON'S STATEMENT WAS BEING UNEXPECTEDLY CONFIRMED
  - hyp: Aaron Grafton's statement was being unexpectedly confirmed.

## Largest Errors

- `2094-142345-0027` `test-clean` WER `0.4`
  - ref: MUNNY I TOULD IKE TO DO INTO DE BARN TO TOMMY TO SEE DE WHITTAWD
  - hyp: Money, I did like to do into the barn to Tommy to see the widod.
- `3997-182399-0012` `test-other` WER `0.3076923076923077`
  - ref: IT WAS ON A LIL OL HOUSE A LIL OL TUMBLE DOWN HOUSE
  - hyp: It was on a little old house, a little old tumble-down house.
- `8188-269290-0051` `test-other` WER `0.3`
  - ref: ANNIE COLCHESTER IS YOUR ROOMFELLOW IS SHE NOT SHE SAID
  - hyp: Any Colchester is your room, fellow. Is she not? She said.
- `3528-168669-0098` `test-other` WER `0.25`
  - ref: THE PRIORESS TOOK BREATH THEN TURNED TO FAUCHELEVENT
  - hyp: The priorist took breath, then turned to Fochlevant.
- `3331-159609-0015` `test-other` WER `0.22727272727272727`
  - ref: I HOPE MARIA BAILEY IS ALL HE THINKS HER SHE ADDED SOFTLY FOR I COULD N'T BEAR TO HAVE HIM DISAPPOINTED AGAIN
  - hyp: I hope Maria Bailey is only thinking," she added softly, for I could not bear to have him disappointed again.
- `5442-41169-0029` `test-other` WER `0.15384615384615385`
  - ref: THERE'S A CLASS INSTINCT TOO OF WHAT ONE OUGHT AND OUGHTN'T TO DO
  - hyp: There's a class instinct too, of what one ought and ought not to do.
- `8280-266249-0030` `test-other` WER `0.15`
  - ref: THEY ARE GAMBLING YONDER AND I'M AFRAID THAT YOUNG FELLOW IS BEING BADLY FLEECED BY THAT MIDDLE AGED MAN OPPOSITE
  - hyp: They're gambling yonder, and I'm afraid that young fellow is being badly fleeced by the middle-aged man opposite.
- `1998-29455-0020` `test-other` WER `0.13333333333333333`
  - ref: THEY WENT ON UP THE HILL AS HAPPY AS ANY ONE NEED WISH TO BE
  - hyp: They went on up the hill as happy as anyone need wish to be.
- `3080-5032-0025` `test-other` WER `0.13043478260869565`
  - ref: I HAVE BEEN STUDYING HOW TOM CHEEKE MIGHT COME BY HIS INTELLIGENCE AND I VERILY BELIEVE HE HAS IT FROM MY COUSIN PETERS
  - hyp: I have been studying how Tom Cheek might come by his intelligence, and I very believe he has it from my cousin Peter's.
- `2609-169640-0010` `test-other` WER `0.10526315789473684`
  - ref: I SOON SAW BOTH PROAS AND GLAD ENOUGH WAS I TO PERCEIVE THAT THEY HAD NOT APPROACHED MATERIALLY NEARER
  - hyp: I soon saw both prats, and grat enough was I to perceive that they had not approached materially nearer.

