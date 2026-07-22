# Manuscript build notes

main.tex is self-contained apart from standard LaTeX packages and references.bib. The local workshop2027.sty deliberately provides a neutral, compact two-column workshop layout because the official Interspeech 2027 workshop template was not available when this artifact was written. It does not claim to be the organizer-issued style file; when the venue releases its class, replace only the class/style layer while retaining the complete manuscript body and bibliography.

After running the experiment pipeline and filling the marked values, build with:

~~~bash
latexmk -pdf main.tex
~~~

The manuscript conditionally includes the three figures emitted into generated/. Before the experiment is run, framed placeholders make the missing files explicit rather than silently omitting them.

generated/RESULTS_TO_PASTE.md and generated/placeholder_values.csv are created by the pipeline. They map the central numeric tags to audited source tables. They do not rewrite the paper automatically: a human must inspect and replace each \placeholder{...} tag.
