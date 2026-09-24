from langchain_community.tools import ShellTool

search_tool = ShellTool()

results = search_tool.invoke('whoami')
print(results)