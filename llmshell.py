#!/usr/bin/env python3

from PyShell import PyShell

class LLMInteractiveShell:
    def __init__(self, history_file="~/.llmhistory", history_len=5000):
        """Initialize the interactive LLM shell with history + tab completion."""
        self.shell = PyShell(history_file=history_file, history_len=history_len)
        self.shell.load_history()
        self.shell.enable_tab_completion()
        self.sources = True

    def run(self):
        """Start the interactive REPL loop."""
        print("type 'exit' or 'quit' to quit\n")

        while True:
            try:
                query = input("% ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not query.strip():
                continue

            query = query.lower()

            # Save command in shell history
            #self.shell.push_input(query)

            # Exit condition
            if query in ("exit", "quit"):
                print("Goodbye!")
                break

            # Execute the query
            try:
                results = self.shell.execute_inputs(query)
            except Exception as e:
                print("LLM Shell Caught: ", e)

            # Show the main answer
            if results and "answer" in results:
                print(results["answer"] + "\n")

            # Show sources if available
            if self.sources:
                print("Sources:")
                for line in results["sources"]:
                    print("\t" + line)


if __name__ == "__main__":
    LLMInteractiveShell().run()
