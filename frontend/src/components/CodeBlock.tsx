// Lightweight regex-based Python/Triton syntax highlighting (no deps).

const KEYWORDS = new Set(
  `def return if else elif for while in and or not is None True False import
   from as with pass break continue lambda class raise try except finally
   yield global nonlocal assert del`.split(/\s+/),
);

type Tok = { text: string; cls: string };

function tokenizeLine(line: string): Tok[] {
  const out: Tok[] = [];
  const re =
    /(#.*$)|("""[\s\S]*?"""|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(@[A-Za-z_][\w.]*)|\b(\d+\.?\d*(?:e[+-]?\d+)?)\b|\b([A-Za-z_]\w*)\b|(\S)|(\s+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(line)) !== null) {
    const [, comment, str, deco, num, word, sym, ws] = m;
    if (comment != null) out.push({ text: comment, cls: "text-slate-500 italic" });
    else if (str != null) out.push({ text: str, cls: "text-amber-300" });
    else if (deco != null) out.push({ text: deco, cls: "text-cyan-300" });
    else if (num != null) out.push({ text: num, cls: "text-violet-300" });
    else if (word != null)
      out.push({
        text: word,
        cls: KEYWORDS.has(word)
          ? "text-emerald-400 font-semibold"
          : word === word.toUpperCase() && word.length > 1
            ? "text-orange-300"
            : "text-slate-200",
      });
    else if (sym != null) out.push({ text: sym, cls: "text-slate-400" });
    else if (ws != null) out.push({ text: ws, cls: "" });
  }
  return out;
}

export default function CodeBlock({ code }: { code: string }) {
  const lines = code.replace(/\n$/, "").split("\n");
  return (
    <div className="overflow-x-auto rounded-xl border border-slate-800 bg-[#070b14] p-4 font-mono text-[13px] leading-6">
      <table className="w-full border-collapse">
        <tbody>
          {lines.map((line, i) => (
            <tr key={i}>
              <td className="select-none pr-4 text-right align-top text-slate-700">{i + 1}</td>
              <td className="whitespace-pre">
                {tokenizeLine(line).map((t, j) => (
                  <span key={j} className={t.cls}>
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
