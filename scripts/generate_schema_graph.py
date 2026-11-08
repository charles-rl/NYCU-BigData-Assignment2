import os

# Common default install paths for Graphviz on Windows
graphviz_paths = [
    r"C:\Program Files\Graphviz\bin",
    r"C:\Program Files (x86)\Graphviz\bin",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Graphviz\bin"),
]

for p in graphviz_paths:
    if os.path.exists(p):
        os.environ["PATH"] += os.pathsep + p
        break

from graphviz import Digraph

# Initialize Digraph
dot = Digraph('GitHub_Schema', format='png')
# Tighten the layout and enforce left-to-right flow
dot.attr(rankdir='LR', splines='spline', nodesep='0.6', ranksep='1.2')
dot.attr('node', shape='none', fontname='Helvetica,Arial,sans-serif', fontsize='11')
dot.attr('edge', fontname='Helvetica,Arial,sans-serif', fontsize='9', color='#4A5568', 
         fontcolor='#2D3748', arrowsize='0.8', penwidth='1.2', arrowhead='vee')

# Define tables
tables = {
    "Users": {"pk": "user_id", "fks": []},
    "Repositories": {"pk": "repo_id", "fks": []},
    "PullRequests": {
        "pk": "pr_id", "fks": [
            ("author_id", "Users", "user_id"),
            ("merged_by_id", "Users", "user_id"),
            ("repo_id", "Repositories", "repo_id")
        ]
    },
    "Issues": {
        "pk": "issue_id", "fks": [
            ("author_id", "Users", "user_id"),
            ("assignee_id", "Users", "user_id"),
            ("pull_request_id", "PullRequests", "pr_id"),
            ("repo_id", "Repositories", "repo_id")
        ]
    },
    "Pushes": {
        "pk": "push_id", "fks": [
            ("user_id", "Users", "user_id"),
            ("repo_id", "Repositories", "repo_id")
        ]
    },
    "PRReviews": {
        "pk": "review_id", "fks": [
            ("reviewer_id", "Users", "user_id"),
            ("pr_id", "PullRequests", "pr_id"),
            ("repo_id", "Repositories", "repo_id")
        ]
    },
    "IssueComments": {
        "pk": "comment_id", "fks": [
            ("author_id", "Users", "user_id"),
            ("issue_id", "Issues", "issue_id"),
            ("repo_id", "Repositories", "repo_id")
        ]
    },
    "PRReviewComments": {
        "pk": "comment_id", "fks": [
            ("author_id", "Users", "user_id"),
            ("pull_request_review_id", "PRReviews", "review_id"),
            ("in_reply_to_id", "PRReviewComments", "comment_id"),
            ("pr_id", "PullRequests", "pr_id"),
            ("repo_id", "Repositories", "repo_id")
        ]
    }
}

# Color Palette
HEADER_COLOR = "#1E3A8A" 
PK_COLOR = "#FEF08A"     
FK_COLOR = "#DBEAFE"     
BORDER_COLOR = "#94A3B8" 

# Function to generate HTML node
def create_node(table_name, data):
    fk_rows = ""
    if not data["fks"]:
        fk_rows = f'<tr><td bgcolor="#F1F5F9" align="center" color="{BORDER_COLOR}" cellpadding="6"><i>No FKs</i></td></tr>'
    else:
        for fk_name, _, _ in data["fks"]:
            fk_rows += f'<tr><td bgcolor="{FK_COLOR}" align="left" port="{fk_name}" color="{BORDER_COLOR}" cellpadding="6"><b>{fk_name}</b> (FK)</td></tr>'

    return f'''<
    <table border="1" cellborder="0" cellspacing="0" cellpadding="6" color="{BORDER_COLOR}" style="rounded">
        <tr><td bgcolor="{HEADER_COLOR}" align="center" cellpadding="8"><font color="white"><b>{table_name}</b></font></td></tr>
        <tr><td bgcolor="{PK_COLOR}" align="left" port="{data["pk"]}" color="{BORDER_COLOR}"><b>🔑 {data["pk"]}</b> (PK)</td></tr>
        {fk_rows}
    </table>>'''

# --- Pipeline Layout using Subgraphs ---

# Layer 1: Core Dimensions
with dot.subgraph(name='cluster_L1') as c1:
    c1.attr(rank='same', style='invis')
    c1.node("Users", create_node("Users", tables["Users"]))
    c1.node("Repositories", create_node("Repositories", tables["Repositories"]))

# Layer 2: Primary Events
with dot.subgraph(name='cluster_L2') as c2:
    c2.attr(rank='same', style='invis')
    c2.node("PullRequests", create_node("PullRequests", tables["PullRequests"]))
    c2.node("Issues", create_node("Issues", tables["Issues"]))
    c2.node("Pushes", create_node("Pushes", tables["Pushes"]))

# Layer 3: Secondary Events
with dot.subgraph(name='cluster_L3') as c3:
    c3.attr(rank='same', style='invis')
    c3.node("PRReviews", create_node("PRReviews", tables["PRReviews"]))
    c3.node("IssueComments", create_node("IssueComments", tables["IssueComments"]))

# Layer 4: Tertiary Events
with dot.subgraph(name='cluster_L4') as c4:
    c4.attr(rank='same', style='invis')
    c4.node("PRReviewComments", create_node("PRReviewComments", tables["PRReviewComments"]))


# Add Edges
for table_name, data in tables.items():
    for fk_name, ref_table, ref_col in data["fks"]:
        if table_name == ref_table:
            # Self-loop: route it out to the left and back into the left
            dot.edge(f"{table_name}:{fk_name}:w", f"{ref_table}:{ref_col}:w", 
                     label=f" {fk_name} ")
        else:
            dot.edge(f"{table_name}:{fk_name}", f"{ref_table}:{ref_col}", 
                     label=f" {fk_name} ")

# Render
output_filename = "schema_graph_pipeline"
dot.render(output_filename, cleanup=True)
print(f"{output_filename}.png created successfully!")