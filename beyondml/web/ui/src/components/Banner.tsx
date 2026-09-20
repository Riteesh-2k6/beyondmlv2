// The exact TUI BANNER string rendered as a <pre> block — character-for-character.
export default function Banner() {
  return (
    <pre
      className="font-mono text-orange leading-none text-[0.42rem] select-none whitespace-pre"
      aria-label="BeyondML"
    >
{`  ██████╗ ███████╗██╗   ██╗ ██████╗ ███╗   ██╗██████╗    ███╗   ███╗██╗     
  ██╔══██╗██╔════╝╚██╗ ██╔╝██╔═══██╗████╗  ██║██╔══██╗   ████╗ ████║██║     
  ██████╔╝█████╗   ╚████╔╝ ██║   ██║██╔██╗ ██║██║  ██║   ██╔████╔██║██║     
  ██╔══██╗██╔══╝    ╚██╔╝  ██║   ██║██║╚██╗██║██║  ██║   ██║╚██╔╝██║██║     
  ██████╔╝███████╗   ██║   ╚██████╔╝██║ ╚████║██████╔╝   ██║ ╚═╝ ██║███████╗
  ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚═════╝    ╚═╝     ╚═╝╚══════╝`}
    </pre>
  )
}
