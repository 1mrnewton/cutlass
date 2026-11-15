title: Captions
description: Write captions from speech recognition.
tools: speech.start, speech.poll, draft.query, draft.apply
keywords: subtitles, subtitle, transcribe, transcript, lyrics, spoken

Run speech recognition with speech.start and speech.poll. Query the draft. Write caption segments in one combo with draft.apply.

1. Find the speech. draft.query "project", then "track <id>" for the lanes with video or audio, gives each clip as id@start+length.
2. Hear it. speech.start "<start_us> <duration_us>" over that stretch of the timeline, then speech.poll "<job>" until it answers words=: one word a line, "<start_us> <end_us> <word>", on the timeline clock. Poll before any edit: an edit makes the job stale and it must be started again.
3. Break the words into lines of three to seven words, at a sentence end or a gap over 400000.
4. Write every line in one draft.apply:
   - add_subtitles puts them on a new text lane (its track_id must not exist yet). Each cue has new segment_id and material_id, target {start, duration} in microseconds on the timeline, and words {start_time, end_time, text} with times in MILLISECONDS from the cue's own start. Cues must not overlap. face is a font file such as Inter[opsz,wght].ttf.
   - add_caption, once a line, ties a caption to the spoken clip (source_id) so it moves and is cut with it; span {start, duration} is the line's stretch inside that clip.
5. Look. set_caption_style gives karaoke, highlight, pop, word-by-word, one-word, neon, bold-box, fade or rise. set_position places it (y -0.7 is low on the frame), set_text_size sizes it.
6. Read one back with draft.query "segment <id>".

draft.query "ops add_subtitles add_caption set_caption_style" shows the exact JSON.
