//! The forward seam: what the executor runs a step command through to get its sampled tokens.
//!
//! One implementation runs the model on the device and samples there; tests stand a fake on the
//! same seam and drive the executor without one. What crosses the seam is tokens, never logits:
//! where the logits live and how a token is drawn from them is the implementation's alone.
//!
//! That implementation is behind the `cuda` feature, so no host gate compiles it. Rust drops a
//! struct's fields in declaration order, so the order it declares them in is the order it tears
//! down in, and reordering them is a change no compiler refuses. The test below reads the
//! declaration out of its source and holds it to the order teardown needs.

use std::error::Error;

use crate::batch::BatchLayout;

/// One model forward and sample per step command.
pub trait Forward {
    type Error: Error + Send + Sync + 'static;

    /// Runs the model over `layout` and samples the rows the layout selected: one token per
    /// selected row, in batch order.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error when the step could not be run; the executor treats
    /// it as fatal.
    fn forward(&mut self, layout: &BatchLayout) -> Result<&[u32], Self::Error>;
}

#[cfg(test)]
mod tests {
    use std::iter::Peekable;

    /// The device implementation's source, read as text because no host gate compiles it. It is
    /// read by hand rather than with a parser dependency, because one declaration in one file is
    /// all this pins; a second struct to pin is the point to take `syn`.
    const DEVICE_FORWARD: &str = include_str!("device/forward.rs");

    /// `CudaForward`'s fields as it declares them. Rust drops a struct's fields in declaration
    /// order, so this is the order the device forward tears down in. The session goes first,
    /// which is the load-bearing part: everything a recording of the capture session bakes an
    /// address into is declared after it, so no graph outlives the memory it reads.
    ///
    /// Every name is listed, not the session alone, because the other pin on this order stops
    /// short of what teardown needs: reordering the declaration without `new`'s literal trips
    /// `clippy::inconsistent_struct_constructor`, which `-D warnings` makes a build failure, but
    /// reordering both together trips nothing, and adding or removing a field trips neither.
    const TEARDOWN_ORDER: [&str; 4] = ["session", "decode_step", "baked", "allocated"];

    /// Reads a string to the `"` that closes it, the caller having read the `"` that opens it.
    /// An escape holds the character after it, so the `"` of a `\"` closes nothing.
    fn read_string(source: &mut impl Iterator<Item = char>) {
        while let Some(character) = source.next() {
            match character {
                '\\' => {
                    source.next();
                }
                '"' => return,
                _ => {}
            }
        }
        panic!("a string closes inside the declaration");
    }

    /// Reads an attribute to the `]` that closes it, the caller having read its `#[`. Brackets
    /// nest inside one, so the depth counts only the ones the attribute itself opens: a string and
    /// a comment inside one are read whole, and so is an attribute spanning lines. A char literal
    /// spells itself as a lifetime does and a raw string holds a `"` that closes nothing, so
    /// neither can be told from what it resembles here; each is a loud failure rather than a
    /// reading that runs on past the `]` and swallows the fields after it.
    fn read_attribute(source: &mut Peekable<impl Iterator<Item = char>>) {
        let mut depth = 1_usize;
        while let Some(character) = source.next() {
            match (character, source.peek().copied()) {
                ('"', _) => read_string(source),
                ('\'', _) => panic!("an attribute holds no char literal"),
                ('r', Some('"' | '#')) => panic!("an attribute holds no raw string"),
                ('/', Some('/')) => {
                    source.next();
                    read_line_comment(source);
                }
                ('/', Some('*')) => {
                    source.next();
                    read_block_comment(source);
                }
                ('[', _) => depth += 1,
                (']', _) => {
                    depth -= 1;
                    if depth == 0 {
                        return;
                    }
                }
                _ => {}
            }
        }
        panic!("an attribute closes inside the declaration");
    }

    /// Reads a line comment to the end of its line, the caller having read its `//`.
    fn read_line_comment(source: &mut impl Iterator<Item = char>) {
        for character in source {
            if character == '\n' {
                return;
            }
        }
    }

    /// Reads a block comment to the `*/` that closes it, the caller having read its `/*`. Rust
    /// nests block comments; one nested here closes at the first `*/`, which reads as a field the
    /// pin rejects rather than as a field that vanishes.
    fn read_block_comment(source: &mut impl Iterator<Item = char>) {
        let mut star = false;
        for character in source {
            if star && character == '/' {
                return;
            }
            star = character == '*';
        }
        panic!("a block comment closes inside the declaration");
    }

    /// `body` with its comments and attributes read out of it, leaving the field declarations and
    /// the commas between them. Each is read to its own close, so nothing a comment or an
    /// attribute holds — a bracket, a comma, a `//`, a newline — reads as part of a field.
    fn without_comments_or_attributes(body: &str) -> String {
        let mut declarations = String::new();
        let mut source = body.chars().peekable();
        while let Some(character) = source.next() {
            match (character, source.peek().copied()) {
                ('#', Some('[')) => {
                    source.next();
                    read_attribute(&mut source);
                }
                ('/', Some('/')) => {
                    source.next();
                    read_line_comment(&mut source);
                }
                ('/', Some('*')) => {
                    source.next();
                    read_block_comment(&mut source);
                }
                _ => declarations.push(character),
            }
        }
        declarations
    }

    /// Whether the `'` at `index` opens a char literal rather than a lifetime: a literal holds
    /// one character and closes on the `'` after it, where a lifetime's name runs on.
    fn is_char_literal(text: &str, index: usize) -> bool {
        let mut rest = text[index..].chars().skip(1);
        matches!((rest.next(), rest.next()), (Some(_), Some('\'')))
    }

    /// `text` split on the commas that separate declarations: the ones outside a type's brackets,
    /// so a comma inside a type starts no declaration.
    ///
    /// Only `(`, `[` and `{` count, because only those three pair in every field type. A `<` can
    /// open nothing — `[u8; 1 << 3]` — and a `>` can close nothing — every `->` — so a depth
    /// counting either is raised by one field's const expression and lowered by the next field's
    /// arrow, and the commas between are read past while the end assert still reads zero. The
    /// cost is that a comma between a type's arguments now starts a declaration of its own:
    /// `HashMap<u32, Slot>` is a loud failure, where the parentheses of `PhantomData<(u32, Slot)>`
    /// or an alias read.
    ///
    /// A `{ … }` const block holds an expression, which this does not read at all: a `<` or a `>`
    /// inside one is a loud failure. The character a char literal holds is read past, so a
    /// bracket it holds opens and closes nothing. A depth still open at the end is a bracket
    /// nothing closed, and every comma after it was read past: a loud failure rather than the
    /// fields after it going missing.
    fn split_declarations(text: &str) -> Vec<&str> {
        let mut declarations = Vec::new();
        let mut depth = 0_usize;
        let mut const_blocks = 0_usize;
        let mut start = 0;
        let mut characters = text.char_indices();
        while let Some((index, character)) = characters.next() {
            match character {
                '\'' if is_char_literal(text, index) => {
                    characters.next();
                }
                '<' | '>' if const_blocks > 0 => {
                    panic!("a const expression in a field type, read: {text}")
                }
                '{' => {
                    const_blocks += 1;
                    depth += 1;
                }
                '}' => {
                    const_blocks = const_blocks.saturating_sub(1);
                    depth = depth.saturating_sub(1);
                }
                '(' | '[' => depth += 1,
                ')' | ']' => depth = depth.saturating_sub(1),
                ',' if depth == 0 => {
                    declarations.push(&text[start..index]);
                    start = index + 1;
                }
                _ => {}
            }
        }
        assert_eq!(depth, 0, "the declarations' brackets close, read: {text}");
        declarations.push(&text[start..]);
        declarations
    }

    /// The fields the struct `name` declares in `source`, in declaration order. A field is read
    /// as a declaration, never as a line: a line bounds neither end of one, and a comment or an
    /// attribute can hold anything a line would be read for. Every shape the reading cannot take
    /// panics, so a declaration this refuses is a declaration to fix, never a field that vanishes
    /// out of an order the pin then passes.
    fn declared_fields(source: &str, name: &str) -> Vec<String> {
        let (_, rest) = source
            .split_once(&format!("struct {name} {{"))
            .unwrap_or_else(|| panic!("the source declares {name}"));
        let body = rest.split_once("\n}").expect("the declaration closes").0;
        split_declarations(&without_comments_or_attributes(body))
            .into_iter()
            .map(str::trim)
            .filter(|declaration| !declaration.is_empty())
            .map(|declaration| {
                let (field, _) = declaration
                    .split_once(':')
                    .unwrap_or_else(|| panic!("a field declaration, read: {declaration}"));
                field.trim().to_owned()
            })
            .collect()
    }

    /// The fields a struct whose body is `body` declares, for holding the reading to a shape the
    /// device forward's own declaration could take.
    fn fields_in(body: &str) -> Vec<String> {
        declared_fields(&format!("struct Fields {{\n{body}\n}}\n"), "Fields")
    }

    #[test]
    fn the_device_forward_declares_its_session_before_the_memory_its_graphs_bake() {
        let fields = declared_fields(DEVICE_FORWARD, "CudaForward");
        assert_eq!(
            fields.first().map(String::as_str),
            Some("session"),
            "`session` is declared first, so the graph set drops before the memory it bakes \
             addresses in. This holds the declaration to that on its own, so no edit to \
             TEARDOWN_ORDER can agree a field ahead of it: {fields:?}"
        );
        assert_eq!(
            fields, TEARDOWN_ORDER,
            "CudaForward's declaration and TEARDOWN_ORDER disagree: fields drop in declaration \
             order, so `session` must come first and every field owning memory a recording bakes \
             must follow it — fix the declaration, or add to TEARDOWN_ORDER a field declared \
             after `session`."
        );
    }

    #[test]
    fn a_line_neither_starts_nor_ends_a_declaration() {
        let body = r"
            session: Replay, comm: Communicator,
            baked
                : BakedAddresses,
            allocated : Allocated
        ";
        assert_eq!(fields_in(body), ["session", "comm", "baked", "allocated"]);
    }

    #[test]
    fn no_attribute_hides_the_field_it_sits_on() {
        let body = r#"
            #[doc = "the ] and the [ and the // are text"] #[allow(unused)] comm: Communicator,
            #[arg(long, default_values_t = [1usize, 8])] buckets: Vec<usize>,
            #[doc = "a 6\" hose"]
            baked: BakedAddresses,
            #[doc = "a 3\" hose"]
            #[cfg_attr(
                feature = "nccl",
                allow(dead_code)
            )]
            session: Replay,
        "#;
        assert_eq!(fields_in(body), ["comm", "buckets", "baked", "session"]);
    }

    #[test]
    fn no_comment_inside_an_attribute_hides_the_field_after_it() {
        let body = r#"
            #[cfg(any(
                // a [ mark
                feature = "nccl"
            ))]
            comm: Communicator,
            #[cfg(/* a ] mark */ any())]
            session: Replay,
        "#;
        assert_eq!(fields_in(body), ["comm", "session"]);
    }

    #[test]
    fn no_comment_reads_as_a_declaration() {
        let body = r"
            // the graph set goes first
            /// Held for the process lifetime.
            /* the graph set * and / the buffers */
            session: Replay, /* then */ comm: Communicator, // and nothing after
        ";
        assert_eq!(fields_in(body), ["session", "comm"]);
    }

    #[test]
    fn a_comma_inside_a_type_starts_no_declaration() {
        let body = r"
            open: fn([u8; 4], u32) -> Replay,
            close: fn(Vec<u8>, u32) -> Replay,
            marker: PhantomData<(u32, Slot)>,
            allocated: Allocated,
        ";
        assert_eq!(fields_in(body), ["open", "close", "marker", "allocated"]);
    }

    #[test]
    fn no_const_expression_hides_the_fields_after_it() {
        let body = r"
            scratch: [u8; 1 << 3],
            comm: Communicator,
            open: fn() -> Replay,
            close: fn() -> Replay,
        ";
        assert_eq!(fields_in(body), ["scratch", "comm", "open", "close"]);
    }

    #[test]
    #[should_panic(expected = "a const expression in a field type, read:")]
    fn a_const_block_in_a_field_type_is_a_loud_failure() {
        fields_in(
            r"
            allocated: Allocated<{ 1 << 2 }>,
            comm: Communicator,
            open: fn() -> Replay,
        ",
        );
    }

    #[test]
    #[should_panic(expected = "a field declaration, read: Slot>")]
    fn a_comma_between_a_types_arguments_is_a_loud_failure() {
        fields_in(
            r"
            slots: HashMap<u32, Slot>,
            session: Replay,
        ",
        );
    }

    #[test]
    fn a_char_literal_holds_a_bracket_that_closes_nothing() {
        let body = r"
            marker: Buffer<{'{'}>,
            slice: &'a [u8],
            slot: Ref<'a>,
            session: Replay,
        ";
        assert_eq!(fields_in(body), ["marker", "slice", "slot", "session"]);
    }

    #[test]
    #[should_panic(expected = "an attribute closes inside the declaration")]
    fn an_attribute_that_never_closes_is_a_loud_failure() {
        fields_in(
            r#"
            #[cfg(feature = "nccl")
            comm: Communicator,
        "#,
        );
    }

    #[test]
    #[should_panic(expected = "a string closes inside the declaration")]
    fn a_string_an_attribute_never_closes_is_a_loud_failure() {
        fields_in(
            r#"
            #[doc = "a 6 hose]
            comm: Communicator,
        "#,
        );
    }

    #[test]
    #[should_panic(expected = "an attribute holds no char literal")]
    fn a_char_literal_in_an_attribute_is_a_loud_failure() {
        fields_in(
            r"
            #[cfg(any('['))]
            comm: Communicator,
            session: Replay,
        ",
        );
    }

    #[test]
    #[should_panic(expected = "an attribute holds no raw string")]
    fn a_raw_string_in_an_attribute_is_a_loud_failure() {
        fields_in(
            r##"
            #[doc = r#"a " mark"#]
            comm: Communicator,
            session: Replay,
        "##,
        );
    }

    #[test]
    #[should_panic(expected = "a block comment closes inside the declaration")]
    fn a_block_comment_that_never_closes_is_a_loud_failure() {
        fields_in(
            r"
            /* the graph set goes first
            session: Replay,
        ",
        );
    }

    #[test]
    #[should_panic(expected = "the declarations' brackets close, read:")]
    fn a_bracket_the_reading_never_closes_is_a_loud_failure() {
        fields_in(
            r"
            marker: Buffer<{
            comm: Communicator,
        ",
        );
    }

    #[test]
    #[should_panic(expected = "a field declaration, read: session Replay")]
    fn text_that_declares_no_field_is_a_loud_failure() {
        fields_in(r"    session Replay,");
    }
}
