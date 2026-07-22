# Literature review and gap derivation

## Search question

The search asked whether no-reference *perceptual speech-quality* scores have been directly validated as fixed-budget selectors of raw human-recorded ASR training utterances, with downstream robustness and retention composition evaluated separately from transcript correctness, speaker mix, content, and duration.

## What the literature establishes

Classical intrusive measures compare a degraded waveform with a clean reference, including PESQ and STOI. Single-ended and learned quality estimators remove the clean-reference requirement: Quality-Net estimates PESQ-like quality, MOSNet predicts voice-conversion MOS, DNSMOS P.835 predicts speech, background, and overall P.835 dimensions for noise suppression, and NISQA predicts overall MOS plus four dimensions for communication impairments. TorchAudio SQUIM supplies reference-free estimates of PESQ, STOI, SI-SDR, and a subjective quality model. These papers establish useful proxy targets; they do not define ASR training utility.

Metric transportability is not automatic. Quality scores and the listener labels used to train them can depend on corpus and evaluation context: Cooper and Yamagishi document range-equalizing bias, Pieper and Voran study the corpus effect and score alignment, and Torcoli et al. find objective-measure reliability depends on application domain. Rossenbach et al. further report no clear relation between NISQA/intelligibility measures and downstream ASR when choosing synthetic TTS sources. This makes downstream validation a necessity rather than a corollary of MOS correlation.

ASR data curation is active but usually optimizes a different signal. GigaSpeech validates transcriptions; ASR quality-estimation work predicts correctness of an already-produced recognition hypothesis; contrastive and discrete-representation methods select domain-relevant audio; and recent pipelines filter pseudo-labels. Quality-driven curation has shown value for speech enhancement and TTS. Those studies are important precedents, but they do not isolate a frozen perceptual score's utility for selecting raw ASR recordings when source utterance, transcript, speaker, duration, and training compute are held invariant.

## Gap and hypothesis

Accordingly, this artifact does **not** claim that speech curation or quality filtering is unexplored. Its narrower contribution is a joint, controlled test of metric choice, fixed-duration selection, source calibration, downstream ASR, and retention composition. The primary counterfactual design tests whether a score can choose between two acoustic renditions of the same spoken utterance. The secondary deployment-style design tests whether global top-score filtering changes source and acoustic coverage. The hypothesis is that quality is conditionally useful within a recording source but an uncalibrated global score is not necessarily a monotonic ranking of ASR training value.

Every bibliographic entry cited by the manuscript was checked against an official proceedings page, publisher record, author-hosted paper, or the paper's own arXiv record before inclusion. `paper/references.bib` contains no unverified entries.
