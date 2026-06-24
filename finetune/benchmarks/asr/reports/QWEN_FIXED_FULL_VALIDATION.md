# Qwen Fixed Full Validation

Run: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_185419_qwen3_asr_1_7b_fixed_text_extraction_baseline_full`

Rows: 5559
Failures: 0
Duplicate IDs: 0
Manifest ID set equal: True
Manifest sorted order equal: False
Malformed hypotheses: 0
Empty hypotheses: 0
Extraction paths: `['$[0].text']`
Result types: `['qwen_asr.inference.qwen3_asr.ASRTranscription']`

## Metrics

Combined normalized WER: 0.02751646508258752
test-clean normalized WER: 0.018411442483262326
test-other normalized WER: 0.03666201784383776
Raw WER: 0.9862560642019081
Normalized CER: 0.009904616200632524
Substitutions: 2337
Deletions: 330
Insertions: 220
Reference words: 104919
Sentence error rate: 0.29807519338010435
Exact-match rate: 0.7019248066198956
Total audio duration sec: 38682.05075
Total inference time sec: 4587.473365384154
Aggregate RTF: 0.1185943680967885
Mean latency sec: 0.8252335609613517
Median latency sec: 0.694949496537447
Peak GPU memory GB: 4.068508625030518

## Random test-clean examples

- `7176-88083-0023` `test-clean` WER `0.0`
  - ref: HE HAD A LOT OF LINE OUT AND THE PLACE WAS NONE TOO FREE FOR A LONG CAST BUT HE WAS IMPATIENT TO DROP HIS FLIES AGAIN ON THE SPOT WHERE THE BIG FISH WAS FEEDING
  - hyp: He had a lot of line out, and the place was none too free for a long cast. But he was impatient to drop his flies again on the spot where the big fish was feeding.
- `5142-36377-0025` `test-clean` WER `0.0`
  - ref: SILAS SLUNK AWAY WITHOUT A WORD OF PROTEST AMBROSE STOOD HIS GROUND EVIDENTLY BENT ON MAKING HIS PEACE WITH NAOMI BEFORE HE LEFT HER SEEING THAT I WAS IN THE WAY I WALKED ASIDE TOWARD A GLASS DOOR AT THE LOWER END OF THE ROOM
  - hyp: Silas slunk away without a word of protest. Ambrose stood his ground, evidently bent on making his peace with Naomi before he left her. Seeing that I was in the way, I walked aside toward a glass door at the lower end of the room.
- `4507-16021-0003` `test-clean` WER `0.0`
  - ref: SHE HAS A SON THEFT AND A DAUGHTER HUNGER
  - hyp: She has a son, theft, and a daughter, hunger.
- `2094-142345-0054` `test-clean` WER `0.07692307692307693`
  - ref: OH SIR SAID MISSUS POYSER RATHER ALARMED YOU WOULDN'T LIKE IT AT ALL
  - hyp: Oh, sir," said Mrs. Poyser, rather alarmed. "You wouldn't like it at all."
- `121-127105-0010` `test-clean` WER `0.0`
  - ref: SHE SENT ME THE PAGES IN QUESTION BEFORE SHE DIED
  - hyp: She sent me the pages in question before she died.
- `2094-142345-0021` `test-clean` WER `0.0`
  - ref: THAT'S THE WAY WITH YOU THAT'S THE ROAD YOU'D ALL LIKE TO GO HEADLONGS TO RUIN
  - hyp: That's the way with you. That's the road you'd all like to go headlongs to ruin.
- `8455-210777-0012` `test-clean` WER `0.2`
  - ref: MISSUS NEVERBEND YOU MUST INDEED BE PROUD OF YOUR SON
  - hyp: Missus never been. You must indeed be proud of your son.
- `1320-122612-0008` `test-clean` WER `0.0`
  - ref: THE EYES OF THE WHOLE PARTY FOLLOWED THE UNEXPECTED MOVEMENT AND READ THEIR SUCCESS IN THE AIR OF TRIUMPH THAT THE YOUTH ASSUMED
  - hyp: The eyes of the whole party followed the unexpected movement and read their success in the air of triumph that the youth assumed.
- `6930-76324-0024` `test-clean` WER `0.16666666666666666`
  - ref: THEY SAY ILLUMINATION BY CANDLE LIGHT IS THE PRETTIEST IN THE WORLD
  - hyp: They say illumination by candlelight is the prettiest in the world.
- `4992-23283-0018` `test-clean` WER `0.0`
  - ref: TO RELIEVE HER FROM BOTH HE LAID HIS HAND WITH FORCE UPON HIS HEART AND SAID DO YOU BELIEVE ME
  - hyp: To relieve her from both, he laid his hand with force upon his heart and said, "Do you believe me?"

## Random test-other examples

- `4852-28311-0014` `test-other` WER `0.10526315789473684`
  - ref: CHRIS STARTED OFF ONCE MORE PASSING THE BLEAK LITTLE VICTORIAN CHURCH PERCHED ON THE HILL ABOVE MISTER WICKER'S HOUSE
  - hyp: Chris started off once more, passing a bleak little Victorian church perched on the hill above Mr. Wicker's house.
- `5442-32873-0005` `test-other` WER `0.07692307692307693`
  - ref: STANLEY STANLEY IT WOULD BE MERCY TO KILL ME SHE BROKE OUT AGAIN
  - hyp: Stanley, Stanley, it would be mercy to kill me. She broke her again.
- `4852-28312-0012` `test-other` WER `0.18181818181818182`
  - ref: JAKEY HARRIS HIS NAME IS AND HE REALLY NEEDS THE JOB
  - hyp: Jakey Harris's name is, and he really needs the job.
- `3528-168669-0031` `test-other` WER `0.0`
  - ref: THE MOTHERS HAVE TAKEN HER TO THE DEAD ROOM WHICH OPENS ON THE CHURCH I KNOW
  - hyp: The mothers have taken her to the dead room, which opens on the church. I know.
- `3331-159609-0008` `test-other` WER `0.0`
  - ref: THANK HEAVEN FOR THAT
  - hyp: Thank heaven for that.
- `5442-32873-0004` `test-other` WER `0.0`
  - ref: OH FRIGHTFUL FRIGHTFUL
  - hyp: Oh, frightful! Frightful!
- `3538-142836-0015` `test-other` WER `0.03225806451612903`
  - ref: MARMALADES AND JAMS DIFFER LITTLE FROM EACH OTHER THEY ARE PRESERVES OF A HALF LIQUID CONSISTENCY MADE BY BOILING THE PULP OF FRUITS AND SOMETIMES PART OF THE RINDS WITH SUGAR
  - hyp: Marmalades and jams differ little from each other. They are preserves of half liquid consistency made by boiling the pulp of fruits, and sometimes part of the rinds, with sugar.
- `3764-168671-0016` `test-other` WER `0.0`
  - ref: JEAN VALJEAN'S COMPOSURE WAS ONE OF THOSE POWERFUL TRANQUILLITIES WHICH ARE CONTAGIOUS
  - hyp: Jean Valjean's composure was one of those powerful tranquillities which are contagious.
- `2414-128292-0024` `test-other` WER `0.0`
  - ref: THOU ART MY SHADOW
  - hyp: Thou art my shadow.
- `3997-180294-0027` `test-other` WER `0.05555555555555555`
  - ref: DID SHE LOVE ME ENOUGH TO BELIEVE THAT THE MORE BEAUTIFUL SHE LOOKED THE HAPPIER I SHOULD BE
  - hyp: Does she love me enough to believe that the more beautiful she looked, the happier I should be?

## Largest-error examples

- `533-131564-0022` `test-other` WER `1.5`
  - ref: WHERE'S MILICENT
  - hyp: Where is Millicent?
- `2609-156975-0016` `test-other` WER `1.375`
  - ref: WAS MOSES JUSTIFIED IN RESISTING THE EGYPTIAN TASKMASTER
  - hyp: Which moves its just fine and resists in the gypsum tacks, master.
- `1089-134691-0024` `test-clean` WER `1.0`
  - ref: STEPHANOS DEDALOS
  - hyp: Stefano Sterlos.
- `4852-28311-0013` `test-other` WER `1.0`
  - ref: AW SHUCKS
  - hyp: Ah, sharks.
- `3538-142836-0023` `test-other` WER `1.0`
  - ref: ICES
  - hyp: Isis.
- `1998-29455-0029` `test-other` WER `0.6666666666666666`
  - ref: CLEVER AS A TRAINDAWG E IS AN ALL OUTER IS OWN EAD
  - hyp: Clever as a train dog, he is, and all out as ownad.
- `3005-163399-0010` `test-other` WER `0.6666666666666666`
  - ref: WHY CHILD IT LL BE STOLE
  - hyp: Watch out! It'll be stole.
- `4198-12259-0030` `test-other` WER `0.6666666666666666`
  - ref: HO THIS WILL BANG IT SOUNDLY
  - hyp: Oh, this was banged soundly.
- `533-131562-0006` `test-other` WER `0.6666666666666666`
  - ref: MISTER HUNTINGDON THEN WENT UP STAIRS
  - hyp: Mr. Huntington then went upstairs.
- `2094-142345-0022` `test-clean` WER `0.6666666666666666`
  - ref: MISTER OTTLEY'S INDEED
  - hyp: Mr. Oatley's, indeed.

## Exact-match examples

- `6432-63723-0000` `test-other` WER `0.0`
  - ref: CHUCKLED THE COLONEL AS HE SKILFULLY PLAYED THE LUCKLESS TROUT NOW STRUGGLING TO GET LOOSE FROM THE HOOK
  - hyp: Chuckled the colonel, as he skilfully played the luckless trout, now struggling to get loose from the hook.
- `7975-280076-0017` `test-other` WER `0.0`
  - ref: OUR BUSINESS THERE WAS TO SEE E P WEST HE WAS NOT AT HOME BUT THE FAMILY WILL REMEMBER THAT WE WERE THERE
  - hyp: Our business there was to see E. P. West. He was not at home, but the family will remember that we were there.
- `8131-117029-0009` `test-other` WER `0.0`
  - ref: THOUGHT YOU'D BE IN THE CHIPS
  - hyp: Thought you'd be in the chips.
- `4350-9170-0049` `test-other` WER `0.0`
  - ref: EXCEPT FOR THE STATE THEY TELL US WE SHOULD NOT HAVE ANY RELIGION EDUCATION CULTURE MEANS OF COMMUNICATION AND SO ON
  - hyp: Except for the state, they tell us we should not have any religion, education, culture, means of communication, and so on.
- `367-130732-0028` `test-other` WER `0.0`
  - ref: PUT THESE INGREDIENTS INTO A STEWPAN AND FRY THEM TEN MINUTES THEN THROW IN THE CRAWFISH AND POUR ON THEM HALF A BOTTLE OF FRENCH WHITE WINE
  - hyp: Put these ingredients into a stewpan and fry them ten minutes. Then throw in the crawfish and pour on them half a bottle of French white wine.
- `1995-1826-0021` `test-clean` WER `0.0`
  - ref: DON'T KNOW WELL OF ALL THINGS INWARDLY COMMENTED MISS TAYLOR LITERALLY BORN IN COTTON AND OH WELL AS MUCH AS TO ASK WHAT'S THE USE SHE TURNED AGAIN TO GO
  - hyp: Don't know. Well, of all things, inwardly commented Miss Taylor, literally born in cotton, and oh well, as much as to ask what's the use. She turned again to go.
- `61-70968-0048` `test-clean` WER `0.0`
  - ref: AND HENRY MIGHT RETURN TO ENGLAND AT ANY MOMENT
  - hyp: And Henry might return to England at any moment.
- `6829-68771-0004` `test-clean` WER `0.0`
  - ref: UNDER ORDINARY CONDITIONS REYNOLDS WAS SURE TO BE ELECTED BUT THE COMMITTEE PROPOSED TO SACRIFICE HIM IN ORDER TO ELECT HOPKINS
  - hyp: Under ordinary conditions, Reynolds was sure to be elected, but the committee proposed to sacrifice him in order to elect Hopkins.
- `1221-135766-0008` `test-clean` WER `0.0`
  - ref: MINDFUL HOWEVER OF HER OWN ERRORS AND MISFORTUNES SHE EARLY SOUGHT TO IMPOSE A TENDER BUT STRICT CONTROL OVER THE INFANT IMMORTALITY THAT WAS COMMITTED TO HER CHARGE
  - hyp: Mindful, however, of her own errors and misfortunes, she early sought to impose a tender but strict control over the infant immortality that was committed to her charge.
- `8188-274364-0001` `test-other` WER `0.0`
  - ref: IN THE GOVERNMENT OF IRELAND HIS ADMINISTRATION HAD BEEN EQUALLY PROMOTIVE OF HIS MASTER'S INTEREST AND THAT OF THE SUBJECTS COMMITTED TO HIS CARE
  - hyp: In the government of Ireland, his administration had been equally promotive of his master's interest, and that of the subjects committed to his care.

