title: Remove pauses
description: Remove pauses and silent gaps from a range.
tools: speech.start, speech.poll, sandbox.run, draft.query, draft.apply

Transcribe the range with speech.start, then speech.poll until the word times are ready. Query the draft with draft.query. Use sandbox.run to find the silent gaps from those word times. Apply the trims that remove those gaps in one combo with draft.apply.

This skill does not write the draft during a frame.
