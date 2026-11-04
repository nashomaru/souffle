/// Rough token estimate without a tokenizer. Transcripts tokenize denser than prose.
///
/// Measured tokens per word on 2000 words of the same French transcript,
/// Q4_K_M: qwen3:4b 1.523, qwen2.5:7b 1.532, llama3.2:3b 1.538, mistral:latest
/// 1.760. Every measured ratio is above 1.4, so that ratio put every budget
/// in the app 9 to 26 percent short of the real token count. This estimate
/// feeds only budget *limits* (stuff-versus-map threshold, reduce batching,
/// the Apple Intelligence budgets), never the chunk size itself, which is a
/// fixed word count. Underestimating causes silent truncation; overestimating
/// only costs a little chunking margin. A single conservative ratio is
/// therefore the right trade: a per-model table would be guesswork for every
/// model not in the measured set.
pub fn estimate_tokens(text: &str) -> usize {
    let word_estimate = (text.split_whitespace().count() as f32 * 1.8).ceil() as usize;
    // Japanese and other CJK transcripts commonly contain no spaces, so a
    // word-only estimate can treat an hour-long transcript as a handful of
    // tokens and incorrectly send it through the single-call path.
    word_estimate.max(text.chars().filter(|character| is_cjk(*character)).count())
}

fn is_cjk(character: char) -> bool {
    matches!(
        character,
        '\u{3040}'..='\u{30ff}'
            | '\u{3400}'..='\u{4dbf}'
            | '\u{4e00}'..='\u{9fff}'
            | '\u{f900}'..='\u{faff}'
            | '\u{ac00}'..='\u{d7af}'
    )
}

/// Approximate the word budget used by the fixed-size chunk configuration.
/// Two CJK characters per word keeps unspaced transcripts conservative
/// without changing the established English/French chunk sizes.
fn chunk_units(text: &str) -> usize {
    let words = text.split_whitespace().count();
    let cjk = text.chars().filter(|character| is_cjk(*character)).count();
    words.max(cjk.div_ceil(2))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkConfig {
    pub stuff_token_limit: usize,
    /// Max estimated tokens for one reduce-stage user prompt (map summaries + framing).
    pub reduce_token_limit: usize,
    pub chunk_words: usize,
    pub chunk_overlap_words: usize,
    pub map_concurrency: usize,
}

impl ChunkConfig {
    pub const OLLAMA: Self = Self {
        stuff_token_limit: 6000,
        reduce_token_limit: 12_000,
        chunk_words: 1400,
        chunk_overlap_words: 120,
        map_concurrency: 2,
    };

    /// Foundation Models ship with a smaller context window than local Ollama.
    /// The 4096-token window covers BOTH prompt and response (Apple TN3193),
    /// so these budgets stay well under it. Apple ships a per-session
    /// `SystemLanguageModel.contextSize` / `tokenCount(for:)` API since macOS
    /// 26.4 for exact, dynamic budgeting instead of this hardcoded estimate;
    /// worth switching to once the real FoundationModels bridge (not the CI
    /// stub) is exercised in this build.
    pub const APPLE_INTELLIGENCE: Self = Self {
        stuff_token_limit: 1500,
        // ~4096 FM context minus system prompt and generation headroom.
        reduce_token_limit: 3_200,
        chunk_words: 450,
        chunk_overlap_words: 40,
        map_concurrency: 1,
    };
}

/// Pack whole turns into overlapping chunks. A turn is never split, even when
/// it is longer than `chunk_words`. Overlap is whole trailing turns whose
/// combined word count fits in `chunk_overlap_words`.
pub fn chunk_turns(turns: &[String], config: ChunkConfig) -> Vec<String> {
    if turns.is_empty() {
        return Vec::new();
    }
    let words = chunk_units;
    let total: usize = turns.iter().map(|turn| words(turn)).sum();
    if total <= config.chunk_words {
        return vec![turns.join("\n")];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < turns.len() {
        let mut end = start;
        let mut count = 0;
        while end < turns.len() {
            let turn_words = words(&turns[end]);
            if count > 0 && count + turn_words > config.chunk_words {
                break;
            }
            count += turn_words;
            end += 1;
        }
        chunks.push(turns[start..end].join("\n"));
        if end >= turns.len() {
            break;
        }
        let mut next = end;
        let mut kept = 0;
        while next > start + 1 {
            let turn_words = words(&turns[next - 1]);
            if kept + turn_words > config.chunk_overlap_words {
                break;
            }
            kept += turn_words;
            next -= 1;
        }
        start = next.max(start + 1);
    }
    chunks
}

/// Split a transcript into overlapping word chunks for the map stage.
pub fn chunk_transcript(text: &str, config: ChunkConfig) -> Vec<String> {
    let cjk_chars = text.chars().filter(|character| is_cjk(*character)).count();
    let whitespace_words = text.split_whitespace().count();
    if cjk_chars.div_ceil(2) > whitespace_words {
        return chunk_unspaced_text(text, config);
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= config.chunk_words {
        return vec![text.to_string()];
    }
    let step = config
        .chunk_words
        .saturating_sub(config.chunk_overlap_words)
        .max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let end = (start + config.chunk_words).min(words.len());
        chunks.push(words[start..end].join(" "));
        if end == words.len() {
            break;
        }
        start += step;
    }
    chunks
}

fn chunk_unspaced_text(text: &str, config: ChunkConfig) -> Vec<String> {
    let characters: Vec<char> = text.chars().collect();
    let chunk_chars = config.chunk_words.saturating_mul(2).max(1);
    if characters.len() <= chunk_chars {
        return vec![text.to_string()];
    }
    let overlap_chars = config.chunk_overlap_words.saturating_mul(2);
    let step = chunk_chars.saturating_sub(overlap_chars).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < characters.len() {
        let end = (start + chunk_chars).min(characters.len());
        chunks.push(characters[start..end].iter().collect());
        if end == characters.len() {
            break;
        }
        start += step;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::{ChunkConfig, chunk_transcript, chunk_turns, chunk_units, estimate_tokens};

    #[test]
    fn estimate_tokens_scales_with_words() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens(&"word ".repeat(10)), 18);
    }

    #[test]
    fn estimate_tokens_counts_unspaced_japanese_text() {
        let text = "本日の会議では新製品の発売日と担当者を決定しました。";
        assert!(estimate_tokens(text) >= text.chars().count() - 2);
    }

    #[test]
    fn short_transcript_is_one_chunk() {
        let text = "word ".repeat(ChunkConfig::OLLAMA.chunk_words);
        assert_eq!(chunk_transcript(&text, ChunkConfig::OLLAMA).len(), 1);
    }

    #[test]
    fn long_transcript_chunks_with_overlap() {
        let total = ChunkConfig::OLLAMA.chunk_words * 3;
        let words: Vec<String> = (0..total).map(|i| i.to_string()).collect();
        let chunks = chunk_transcript(&words.join(" "), ChunkConfig::OLLAMA);
        assert!(
            chunks.len() >= 3,
            "expected multiple chunks, got {}",
            chunks.len()
        );
        let step = ChunkConfig::OLLAMA.chunk_words - ChunkConfig::OLLAMA.chunk_overlap_words;
        assert!(chunks[1].split_whitespace().next().unwrap() == step.to_string());
        assert!(chunks[0].split_whitespace().any(|w| w == step.to_string()));
    }

    #[test]
    fn apple_chunks_are_smaller_than_ollama() {
        let total = ChunkConfig::OLLAMA.chunk_words * 3;
        let words: Vec<String> = (0..total).map(|i| i.to_string()).collect();
        let text = words.join(" ");
        let ollama = chunk_transcript(&text, ChunkConfig::OLLAMA);
        let apple = chunk_transcript(&text, ChunkConfig::APPLE_INTELLIGENCE);
        assert!(apple.len() > ollama.len());
    }

    #[test]
    fn long_japanese_transcript_is_split_for_apple_intelligence() {
        let sentence = "本日の会議では新製品の発売日と担当者を決定しました。";
        let text = sentence.repeat(600);
        let chunks = chunk_transcript(&text, ChunkConfig::APPLE_INTELLIGENCE);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| {
            chunk.chars().count() <= ChunkConfig::APPLE_INTELLIGENCE.chunk_words * 2
        }));
        assert!(chunks.iter().any(|chunk| chunk.contains("発売日")));
    }

    fn tight(chunk_words: usize, overlap: usize) -> ChunkConfig {
        ChunkConfig {
            stuff_token_limit: 6000,
            reduce_token_limit: 12_000,
            chunk_words,
            chunk_overlap_words: overlap,
            map_concurrency: 1,
        }
    }

    #[test]
    fn short_turns_fit_in_one_chunk() {
        let turns = vec![
            "[0:00] Me: hello there".into(),
            "[0:05] Them: hi back".into(),
        ];
        let chunks = chunk_turns(&turns, tight(50, 4));
        assert_eq!(chunks, vec!["[0:00] Me: hello there\n[0:05] Them: hi back"]);
    }

    #[test]
    fn packing_never_splits_a_turn() {
        let long = format!("[0:00] Me: unique-AAA {}", "pad ".repeat(30).trim());
        let turns = vec![long.clone(), "[0:40] Them: unique-BBB".into()];
        let chunks = chunk_turns(&turns, tight(10, 2));
        let with_aaa: Vec<_> = chunks.iter().filter(|c| c.contains("unique-AAA")).collect();
        assert_eq!(with_aaa.len(), 1);
        assert_eq!(*with_aaa[0], long);
        assert!(chunks.iter().any(|c| c.contains("unique-BBB")));
        assert!(
            !chunks
                .iter()
                .any(|c| c.contains("unique-AAA") && c.contains("unique-BBB"))
        );
    }

    #[test]
    fn a_turn_longer_than_the_budget_is_its_own_chunk() {
        let long = format!("[0:00] Me: {}", "word ".repeat(20).trim());
        let chunks = chunk_turns(&[long.clone(), "[1:00] Them: bye".into()], tight(8, 2));
        assert_eq!(chunks[0], long);
        assert_eq!(chunks[1], "[1:00] Them: bye");
    }

    #[test]
    fn overlap_is_whole_trailing_turns() {
        let turns = vec![
            "[0:00] Me: one two three four".into(),
            "[0:10] Them: five six seven eight".into(),
            "[0:20] Me: nine ten eleven twelve".into(),
        ];
        // 6 words/turn; pack two per chunk; overlap one turn (6 <= 8).
        let chunks = chunk_turns(&turns, tight(12, 8));
        assert!(chunks.len() >= 2);
        assert!(chunks[0].contains("[0:00] Me:"));
        assert!(chunks[0].contains("[0:10] Them:"));
        assert!(
            chunks[1].contains("[0:10] Them:"),
            "overlap should keep the trailing turn"
        );
        assert!(chunks[1].contains("[0:20] Me:"));
    }

    #[test]
    fn sixty_minute_japanese_meeting_packs_into_bounded_turn_chunks() {
        let utterance = "新製品の発売日は十二月で、田中さんが資料を準備します。";
        let turns: Vec<String> = (0..360)
            .map(|minute| format!("[{minute}:00] Them: {}", utterance.repeat(4)))
            .collect();
        let chunks = chunk_turns(&turns, ChunkConfig::APPLE_INTELLIGENCE);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| {
            chunk_units(chunk)
                <= ChunkConfig::APPLE_INTELLIGENCE.chunk_words + chunk_units(&turns[0])
        }));
        assert_eq!(
            chunks
                .iter()
                .filter(|chunk| chunk.contains("[359:00] Them:"))
                .count(),
            1
        );
    }
}
