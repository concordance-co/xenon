# Emotion replication data

Checked-in assets are the label/topic lists used by prompt builders: `emotions.txt` and `topics.txt`. Generated stories, dialogues and captures are stored outside this directory.

For the Llama generation workflow, `generated_rows_table()` resolves `EMOTION_ASSET_GENERATED_ROWS_TABLE`, defaulting to `papers_voice_emotions_llama70b_generated_rows_v1`. The vector workflow consumes the configured prepared rows. Other recipes may use different tables or artifact-backed datasets; inspect the chosen workflow rather than assuming this default applies.

Before continuing a study, record its actual table/query, row count and split, generator/model revision, parsing/filtering policy and source run IDs in the study's phase record. Discover captures and derived vectors through the catalog and asset manifest. This directory is not a live run ledger or a place to commit generated data.

See the [workflow index](../README.md) and [locality rules](../../../../../operations/LOCALITY.md).
