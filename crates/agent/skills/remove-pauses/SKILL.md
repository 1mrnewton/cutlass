title: Remove pauses
description: Remove pauses and silent gaps from a range.
tools: speech.start, speech.poll, sandbox.run, draft.query, draft.apply
keywords: silence, silences, quiet, dead air, jump cut, tighten

Transcribe the range with speech.start, then speech.poll until the word times are ready. Query the draft with draft.query. Use sandbox.run to find the silent gaps from those word times. Apply the trims that remove those gaps in one combo with draft.apply.

1. draft.query "track <id>" for the lane the speech is on: its clips as id@start+length.
2. speech.start "<start_us> <duration_us>" over the range, then speech.poll until words=: one word a line, "<start_us> <end_us> <word>", on the timeline clock. Poll before any edit, as an edit makes the job stale.
3. A pause is the stretch from one word's end to the next word's start when it is longer than the threshold (500000 unless the user names one). Keep 100000 of air on each side of it. sandbox.run can list the pauses: give it the word lines as input.
4. Cut each pause out, from the LAST pause to the first so the earlier times stay true: split the clip at the pause's start (right_id a new id), split that right piece at the pause's end (another new id), and remove the piece between. On the main track the magnet closes the gap; on another lane move the later clips left by the pause's length.
5. Put every split and remove in one draft.apply, so it is one undo step. With many pauses, send it with dry_run true first.
6. draft.query the track again and say how many pauses went and how much shorter the cut is.

This skill does not write the draft during a frame.
