# Qwen Old Run Extraction Audit

Old run: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_090118_qwen3_asr_1_7b_baseline_full`
Predictions: `/home/hj/Data_Collect_Web/finetune/benchmarks/asr/runs/20260624_090118_qwen3_asr_1_7b_baseline_full/predictions.jsonl`
Rows: 5559
Classification: **structured_object_repr_containing_transcript**

## Repr Pattern Counts

- ASRResult(: 0
- TranscriptionResult(: 0
- language_equals: 5559
- text_equals: 5559
- object_at: 0
- class_repr_prefix: 5559

Flagged rows: 5559

## Interpretation

Stored hypotheses include structured result object reprs or embedded language/text fields; old WER is invalid for reporting.

## Sampled Rows

### First/Middle/Last

- id `1089-134686-0000` split `test-clean` status `success` WER `0.25` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE
  - hypothesis: ASRTranscription(language='English', text='He hoped there would be stew for dinner, turnips and carrots, and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.', time_stamps
  - normalized_reference: he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce
  - normalized_hypothesis: asrtranscription language english text he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce time stamps none
- id `1089-134686-0001` split `test-clean` status `success` WER `0.875` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: STUFF IT INTO YOU HIS BELLY COUNSELLED HIM
  - hypothesis: ASRTranscription(language='English', text='Stuff it into you," his belly counselled him.', time_stamps=None)
  - normalized_reference: stuff it into you his belly counselled him
  - normalized_hypothesis: asrtranscription language english text stuff it into you his belly counselled him time stamps none
- id `1089-134686-0002` split `test-clean` status `success` WER `0.3888888888888889` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
  - hypothesis: ASRTranscription(language='English', text='After early nightfall, the yellow lamps would light up here and there the squalid quarter of the brothels.', time_stamps=None)
  - normalized_reference: after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels
  - normalized_hypothesis: asrtranscription language english text after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels time stamps none
- id `4852-28312-0027` split `test-other` status `success` WER `0.3684210526315789` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: A COURTYARD WAS SPARSELY LIT BY A FLARING TORCH OR TWO SHOWING A SWINGING SIGN HUNG ON A POST
  - hypothesis: ASRTranscription(language='English', text='A courtyard was sparsely lit by a flaring torch or two, showing a swinging sign hung on a post.', time_stamps=None)
  - normalized_reference: a courtyard was sparsely lit by a flaring torch or two showing a swinging sign hung on a post
  - normalized_hypothesis: asrtranscription language english text a courtyard was sparsely lit by a flaring torch or two showing a swinging sign hung on a post time stamps none
- id `4852-28312-0028` split `test-other` status `success` WER `0.4375` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: THE POST WAS PLANTED AT THE EDGE OF WHAT WAS NOW A BROAD AND MUDDY ROAD
  - hypothesis: ASRTranscription(language='English', text='The post was planted at the edge of what was now a broad and muddy road.', time_stamps=None)
  - normalized_reference: the post was planted at the edge of what was now a broad and muddy road
  - normalized_hypothesis: asrtranscription language english text the post was planted at the edge of what was now a broad and muddy road time stamps none
- id `4852-28312-0029` split `test-other` status `success` WER `0.4117647058823529` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: A COACH WITH ITS TOP PILED HIGH WITH LUGGAGE STAMPED TO A HALT BESIDE THE FLAGGED COURTYARD
  - hypothesis: ASRTranscription(language='English', text='A coach with its top piled high with luggage stamped to a halt beside the flagged courtyard.', time_stamps=None)
  - normalized_reference: a coach with its top piled high with luggage stamped to a halt beside the flagged courtyard
  - normalized_hypothesis: asrtranscription language english text a coach with its top piled high with luggage stamped to a halt beside the flagged courtyard time stamps none
- id `908-31957-0023` split `test-clean` status `success` WER `0.3888888888888889` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: I LOVE THEE FREELY AS MEN STRIVE FOR RIGHT I LOVE THEE PURELY AS THEY TURN FROM PRAISE
  - hypothesis: ASRTranscription(language='English', text='I love thee freely, as men strive for right. I love thee purely, as they turn from praise.', time_stamps=None)
  - normalized_reference: i love thee freely as men strive for right i love thee purely as they turn from praise
  - normalized_hypothesis: asrtranscription language english text i love thee freely as men strive for right i love thee purely as they turn from praise time stamps none
- id `908-31957-0024` split `test-clean` status `success` WER `0.4444444444444444` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: I LOVE THEE WITH THE PASSION PUT TO USE IN MY OLD GRIEFS AND WITH MY CHILDHOOD'S FAITH
  - hypothesis: ASRTranscription(language='English', text="I love thee with the passion put to use, and my old griefs, and with my childhood's faith.", time_stamps=None)
  - normalized_reference: i love thee with the passion put to use in my old griefs and with my childhood's faith
  - normalized_hypothesis: asrtranscription language english text i love thee with the passion put to use and my old griefs and with my childhood's faith time stamps none
- id `908-31957-0025` split `test-clean` status `success` WER `0.2631578947368421` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: I LOVE THEE WITH A LOVE I SEEMED TO LOSE WITH MY LOST SAINTS I LOVE THEE WITH THE BREATH SMILES TEARS OF ALL MY LIFE AND IF GOD CHOOSE I SHALL BUT LOVE THEE BETTER AFTER DEATH
  - hypothesis: ASRTranscription(language='English', text='I loved thee with the love I seemed to lose with my lost saints. I loved thee with the breath, smiles, tears of all my life, and, if God choose, I shall but love thee better aft
  - normalized_reference: i love thee with a love i seemed to lose with my lost saints i love thee with the breath smiles tears of all my life and if god choose i shall but love thee better after death
  - normalized_hypothesis: asrtranscription language english text i loved thee with the love i seemed to lose with my lost saints i loved thee with the breath smiles tears of all my life and if god choose i shall but love thee better after death t

### Test-clean

- id `1089-134686-0000` split `test-clean` status `success` WER `0.25` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE
  - hypothesis: ASRTranscription(language='English', text='He hoped there would be stew for dinner, turnips and carrots, and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.', time_stamps
  - normalized_reference: he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce
  - normalized_hypothesis: asrtranscription language english text he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce time stamps none
- id `1089-134686-0001` split `test-clean` status `success` WER `0.875` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: STUFF IT INTO YOU HIS BELLY COUNSELLED HIM
  - hypothesis: ASRTranscription(language='English', text='Stuff it into you," his belly counselled him.', time_stamps=None)
  - normalized_reference: stuff it into you his belly counselled him
  - normalized_hypothesis: asrtranscription language english text stuff it into you his belly counselled him time stamps none
- id `1089-134686-0002` split `test-clean` status `success` WER `0.3888888888888889` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
  - hypothesis: ASRTranscription(language='English', text='After early nightfall, the yellow lamps would light up here and there the squalid quarter of the brothels.', time_stamps=None)
  - normalized_reference: after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels
  - normalized_hypothesis: asrtranscription language english text after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels time stamps none
- id `4970-29095-0005` split `test-clean` status `success` WER `0.7` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: THY WAYS GREATLY TRY ME RUTH AND ALL THY RELATIONS
  - hypothesis: ASRTranscription(language='English', text='Thy ways greatly try me, Ruth, and all thy relations.', time_stamps=None)
  - normalized_reference: thy ways greatly try me ruth and all thy relations
  - normalized_hypothesis: asrtranscription language english text thy ways greatly try me ruth and all thy relations time stamps none
- id `908-31957-0025` split `test-clean` status `success` WER `0.2631578947368421` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: I LOVE THEE WITH A LOVE I SEEMED TO LOSE WITH MY LOST SAINTS I LOVE THEE WITH THE BREATH SMILES TEARS OF ALL MY LIFE AND IF GOD CHOOSE I SHALL BUT LOVE THEE BETTER AFTER DEATH
  - hypothesis: ASRTranscription(language='English', text='I loved thee with the love I seemed to lose with my lost saints. I loved thee with the breath, smiles, tears of all my life, and, if God choose, I shall but love thee better aft
  - normalized_reference: i love thee with a love i seemed to lose with my lost saints i love thee with the breath smiles tears of all my life and if god choose i shall but love thee better after death
  - normalized_hypothesis: asrtranscription language english text i loved thee with the love i seemed to lose with my lost saints i loved thee with the breath smiles tears of all my life and if god choose i shall but love thee better after death t

### Test-other

- id `1688-142285-0000` split `test-other` status `success` WER `0.21875` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: THERE'S IRON THEY SAY IN ALL OUR BLOOD AND A GRAIN OR TWO PERHAPS IS GOOD BUT HIS HE MAKES ME HARSHLY FEEL HAS GOT A LITTLE TOO MUCH OF STEEL ANON
  - hypothesis: ASRTranscription(language='English', text="There's iron, they say, in all our blood, and a grain or two, perhaps, is good, but his—he makes me harshly feel—has got a little too much of steel. Anon.", time_stamps=None)
  - normalized_reference: there's iron they say in all our blood and a grain or two perhaps is good but his he makes me harshly feel has got a little too much of steel anon
  - normalized_hypothesis: asrtranscription language english text there's iron they say in all our blood and a grain or two perhaps is good but his he makes me harshly feel has got a little too much of steel anon time stamps none
- id `1688-142285-0001` split `test-other` status `success` WER `0.29411764705882354` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: MARGARET SAID MISTER HALE AS HE RETURNED FROM SHOWING HIS GUEST DOWNSTAIRS I COULD NOT HELP WATCHING YOUR FACE WITH SOME ANXIETY WHEN MISTER THORNTON MADE HIS CONFESSION OF HAVING BEEN A SHOP BOY
  - hypothesis: ASRTranscription(language='English', text='Margaret said, "Mr. Hale, as he returned from showing his guests downstairs, I could not help watching your face with some anxiety, when Mr. Thornton made his confession of havi
  - normalized_reference: margaret said mister hale as he returned from showing his guest downstairs i could not help watching your face with some anxiety when mister thornton made his confession of having been a shop boy
  - normalized_hypothesis: asrtranscription language english text margaret said mr hale as he returned from showing his guests downstairs i could not help watching your face with some anxiety when mr thornton made his confession of having been a s
- id `1688-142285-0002` split `test-other` status `success` WER `0.7777777777777778` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: YOU DON'T MEAN THAT YOU THOUGHT ME SO SILLY
  - hypothesis: ASRTranscription(language='English', text="You don't mean that. You thought me so silly.", time_stamps=None)
  - normalized_reference: you don't mean that you thought me so silly
  - normalized_hypothesis: asrtranscription language english text you don't mean that you thought me so silly time stamps none
- id `4852-28311-0026` split `test-other` status `success` WER `0.7777777777777778` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: A SUDDEN CAR HORN WOKE HIM FROM HIS DREAM
  - hypothesis: ASRTranscription(language='English', text='A sudden car horn woke him from his dream.', time_stamps=None)
  - normalized_reference: a sudden car horn woke him from his dream
  - normalized_hypothesis: asrtranscription language english text a sudden car horn woke him from his dream time stamps none
- id `8461-281231-0038` split `test-other` status `success` WER `0.32` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: THE PRECEPTORS OF WHOM THERE WERE FOUR PRESENT OCCUPIED SEATS BEHIND THEIR SUPERIORS AND BEHIND THEM STOOD THE ESQUIRES OF THE ORDER ROBED IN WHITE
  - hypothesis: ASRTranscription(language='English', text='The preceptors, of whom there were four present, occupied seats behind the superiors, and behind them stood the esquires of the order, robed in white.', time_stamps=None)
  - normalized_reference: the preceptors of whom there were four present occupied seats behind their superiors and behind them stood the esquires of the order robed in white
  - normalized_hypothesis: asrtranscription language english text the preceptors of whom there were four present occupied seats behind the superiors and behind them stood the esquires of the order robed in white time stamps none

### Largest Errors

- id `3538-142836-0023` split `test-other` status `success` WER `8.0` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: ICES
  - hypothesis: ASRTranscription(language='English', text='Isis.', time_stamps=None)
  - normalized_reference: ices
  - normalized_hypothesis: asrtranscription language english text isis time stamps none
- id `2094-142345-0041` split `test-clean` status `success` WER `7.0` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: DIRECTION
  - hypothesis: ASRTranscription(language='English', text='Direction.', time_stamps=None)
  - normalized_reference: direction
  - normalized_hypothesis: asrtranscription language english text direction time stamps none
- id `2414-128291-0020` split `test-other` status `success` WER `7.0` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: WELL
  - hypothesis: ASRTranscription(language='English', text='Well.', time_stamps=None)
  - normalized_reference: well
  - normalized_hypothesis: asrtranscription language english text well time stamps none
- id `7902-96592-0020` split `test-other` status `success` WER `7.0` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: NONSENSE
  - hypothesis: ASRTranscription(language='English', text='Nonsense.', time_stamps=None)
  - normalized_reference: nonsense
  - normalized_hypothesis: asrtranscription language english text nonsense time stamps none
- id `8555-292519-0002` split `test-clean` status `success` WER `7.0` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: VENICE
  - hypothesis: ASRTranscription(language='English', text='Venice.', time_stamps=None)
  - normalized_reference: venice
  - normalized_hypothesis: asrtranscription language english text venice time stamps none
- id `533-131564-0022` split `test-other` status `success` WER `5.0` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: WHERE'S MILICENT
  - hypothesis: ASRTranscription(language='English', text='Where is Millicent?', time_stamps=None)
  - normalized_reference: where's milicent
  - normalized_hypothesis: asrtranscription language english text where is millicent time stamps none
- id `1089-134691-0024` split `test-clean` status `success` WER `4.5` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: STEPHANOS DEDALOS
  - hypothesis: ASRTranscription(language='English', text='Stefano Sterlos.', time_stamps=None)
  - normalized_reference: stephanos dedalos
  - normalized_hypothesis: asrtranscription language english text stefano sterlos time stamps none
- id `4852-28311-0013` split `test-other` status `success` WER `4.5` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: AW SHUCKS
  - hypothesis: ASRTranscription(language='English', text='Ah, sharks.', time_stamps=None)
  - normalized_reference: aw shucks
  - normalized_hypothesis: asrtranscription language english text ah sharks time stamps none

### Flagged

- id `1089-134686-0000` split `test-clean` status `success` WER `0.25` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE
  - hypothesis: ASRTranscription(language='English', text='He hoped there would be stew for dinner, turnips and carrots, and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.', time_stamps
  - normalized_reference: he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce
  - normalized_hypothesis: asrtranscription language english text he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce time stamps none
- id `1089-134686-0001` split `test-clean` status `success` WER `0.875` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: STUFF IT INTO YOU HIS BELLY COUNSELLED HIM
  - hypothesis: ASRTranscription(language='English', text='Stuff it into you," his belly counselled him.', time_stamps=None)
  - normalized_reference: stuff it into you his belly counselled him
  - normalized_hypothesis: asrtranscription language english text stuff it into you his belly counselled him time stamps none
- id `1089-134686-0002` split `test-clean` status `success` WER `0.3888888888888889` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
  - hypothesis: ASRTranscription(language='English', text='After early nightfall, the yellow lamps would light up here and there the squalid quarter of the brothels.', time_stamps=None)
  - normalized_reference: after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels
  - normalized_hypothesis: asrtranscription language english text after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels time stamps none
- id `1089-134686-0003` split `test-clean` status `success` WER `1.0` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: HELLO BERTIE ANY GOOD IN YOUR MIND
  - hypothesis: ASRTranscription(language='English', text='Hello, Bertie. Any good in your mind?', time_stamps=None)
  - normalized_reference: hello bertie any good in your mind
  - normalized_hypothesis: asrtranscription language english text hello bertie any good in your mind time stamps none
- id `1089-134686-0004` split `test-clean` status `success` WER `0.7272727272727273` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND
  - hypothesis: ASRTranscription(language='English', text='Number ten, fresh Nellie is waiting on you. Good night, husband.', time_stamps=None)
  - normalized_reference: number ten fresh nelly is waiting on you good night husband
  - normalized_hypothesis: asrtranscription language english text number ten fresh nellie is waiting on you good night husband time stamps none
- id `1089-134686-0005` split `test-clean` status `success` WER `0.3181818181818182` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: THE MUSIC CAME NEARER AND HE RECALLED THE WORDS THE WORDS OF SHELLEY'S FRAGMENT UPON THE MOON WANDERING COMPANIONLESS PALE FOR WEARINESS
  - hypothesis: ASRTranscription(language='English', text="The music came nearer, and he recalled the words, the words of Shelley's fragment upon the moon, wandering companionless, pale for weariness.", time_stamps=None)
  - normalized_reference: the music came nearer and he recalled the words the words of shelley's fragment upon the moon wandering companionless pale for weariness
  - normalized_hypothesis: asrtranscription language english text the music came nearer and he recalled the words the words of shelley's fragment upon the moon wandering companionless pale for weariness time stamps none
- id `1089-134686-0006` split `test-clean` status `success` WER `0.2916666666666667` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: THE DULL LIGHT FELL MORE FAINTLY UPON THE PAGE WHEREON ANOTHER EQUATION BEGAN TO UNFOLD ITSELF SLOWLY AND TO SPREAD ABROAD ITS WIDENING TAIL
  - hypothesis: ASRTranscription(language='English', text='The dull light fell more faintly upon the page, whereon another equation began to unfold itself slowly, and to spread abroad its widening tail.', time_stamps=None)
  - normalized_reference: the dull light fell more faintly upon the page whereon another equation began to unfold itself slowly and to spread abroad its widening tail
  - normalized_hypothesis: asrtranscription language english text the dull light fell more faintly upon the page whereon another equation began to unfold itself slowly and to spread abroad its widening tail time stamps none
- id `1089-134686-0007` split `test-clean` status `success` WER `0.875` flags `['language_equals', 'text_equals', 'class_repr_prefix']`
  - reference: A COLD LUCID INDIFFERENCE REIGNED IN HIS SOUL
  - hypothesis: ASRTranscription(language='English', text='A cold, lucid indifference reigned in his soul.', time_stamps=None)
  - normalized_reference: a cold lucid indifference reigned in his soul
  - normalized_hypothesis: asrtranscription language english text a cold lucid indifference reigned in his soul time stamps none

