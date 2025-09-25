import Editor from "@monaco-editor/react";

export function CodeEditor({
                               value,
                               onChange,
                           }: {
    value: string;
    onChange: (v: string) => void;
}) {
    return (
        <Editor
            height="70vh"
            defaultLanguage="ruby" // fallback: switch to "plaintext" if ruby isn’t highlighted
            value={value}
            onChange={(v) => onChange(v || "")}
            theme="vs-dark"
            options={{
                minimap: { enabled: false },
                fontSize: 14,
                scrollBeyondLastLine: false,
                wordWrap: "on",
            }}
        />
    );
}
