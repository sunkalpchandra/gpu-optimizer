// Lightweight regex-based Python/Triton syntax highlighting (no deps).

const KEYWORDS = new Set(
  `def return if else elif for while in and or not is None True False import
   from as with pass break continue lambda class raise try except finally
   yield global nonlocal assert del`.split(/\s+/),
);

type Tok = { text: string; color: string };

const C = {
  comment: "#7d8799",
  string: "#a5d6ff",
  decorator: "#d2a8ff",
  number: "#79c0ff",
  keyword: "#ff7b72",
  constexpr: "#ffa657",
  plain: "#c9d1e0",
  punct: "#8b949e",
};

function tokenizeLine(line: string): Tok[] {
  const out: Tok[] = [];
  const re =
    /(#.*$)|("""[\s\S]*?"""|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(@[A-Za-z_][\w.]*)|\b(\d+\.?\d*(?:e[+-]?\d+)?)\b|\b([A-Za-z_]\w*)\b|(\S)|(\s+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(line)) !== null) {
    const [, comment, str, deco, num, word, sym, ws] = m;
    if (comment != null) out.push({ text: comment, color: C.comment });
    else if (str != null) out.push({ text: str, color: C.string });
    else if (deco != null) out.push({ text: deco, color: C.decorator });
    else if (num != null) out.push({ text: num, color: C.number });
    else if (word != null)
      out.push({
        text: word,
        color: KEYWORDS.has(word)
          ? C.keyword
          : word === word.toUpperCase() && word.length > 1
            ? C.constexpr
            : C.plain,
      });
    else if (sym != null) out.push({ text: sym, color: C.punct });
    else if (ws != null) out.push({ text: ws, color: C.plain });
  }
  return out;
}

export default function CodeBlock({ code }: { code: string }) {
  const lines = code.replace(/\n$/, "").split("\n");
  return (
    <div className="overflow-x-auto p-3 font-mono text-[12.5px] leading-[1.55]"
         style={{ background: "#0a0d13" }}>
      <table className="border-collapse">
        <tbody>
          {lines.map((line, i) => (
            <tr key={i}>
              <td className="select-none pr-4 text-right align-top" style={{ color: "#39414f" }}>
                {i + 1}
              </td>
              <td className="whitespace-pre">
                {tokenizeLine(line).map((t, j) => (
                  <span key={j} style={{ color: t.color, fontStyle: t.color === C.comment ? "italic" : undefined }}>
                    {t.text}
                  </span>
                ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
