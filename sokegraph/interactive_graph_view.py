import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
import os

class InteractiveGraphView:
    _FOCUS = "focus_node"

    def __init__(self, kg):
        self.kg = kg  # Your Neo4jKnowledgeGraph instance

    def render(self):
        st.subheader("Interactive Knowledge‑Graph")

        # ── Select node type (label)
        node_labels = self.kg.get_node_labels()
        selected_label = st.selectbox("Select node type", node_labels)

        # ── Select value (e.g., name of a Layer/Keyword/etc.)
        values = self.kg.get_attr_values(selected_label)
        choice = st.selectbox("Value", values, key="attr_value")

        focus = st.session_state.get(self._FOCUS)

        # # ── Get subgraph from KG
        # if focus:
        #     G = self.kg.neighbour_subgraph(focus)
        #     st.markdown(f"##### Neighbours of **{focus}**")
        # else:
        #     G = self.kg.subgraph_for_attr(selected_label, current_val)
        #     st.markdown(
        #         f"##### Nodes where **{selected_label} = {current_val}** &nbsp; "
        #         "(click a node to zoom)"
        #     )

        if choice:
            data = fetch_related(driver, choice)
            st.subheader(f"Neighbours of **{choice}**")
            st.dataframe(data, use_container_width=True)

            if st.checkbox("Visualise neighbours (PyVis)"):
                net = Network(height="500px", width="100%", directed=True)
                style = dict(shape="circle", font={"color": "#ffffff", "size": 14}, color={"background": "#FF5733", "border": "#FF5733"}, borderWidth=2)
                net.add_node(choice, label=wrap_label(choice), **style)

                for row in data:
                    n = row["neighbour"]
                    rel = row["rel_type"]
                    if n not in net.node_map:
                        net.add_node(n, label=wrap_label(n), **style)
                    if row["direction"] == "out":
                        net.add_edge(choice, n, label=rel)
                    else:
                        net.add_edge(n, choice, label=rel)
                net.repulsion(node_distance=180, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)
                html = net.generate_html()
                st.components.v1.html(html, height=800, scrolling=True)

    def _render_pyvis_graph(self, G):
        st.markdown("### 🔍 Debug Info")
        st.write("✅ Nodes in G:", len(G.nodes))
        st.write("✅ Edges in G:", len(G.edges))

        if len(G.nodes) == 0:
            st.error("❌ The subgraph has **no nodes**.")
            return
        if len(G.edges) == 0:
            st.warning("⚠️ The subgraph has **nodes but no edges**.")
        net = Network(height="600px", width="100%", directed=True, notebook=False)
        net.repulsion()  # Better layout

        # Add nodes
        for node_id, data in G.nodes(data=True):
            net.add_node(
                str(node_id),
                label=data.get("name", str(node_id)),
                title=", ".join(data.get("labels", [])) if "labels" in data else str(node_id),
                color="lightblue"
            )

        # Add edges
        for source, target, data in G.edges(data=True):
            net.add_edge(
                str(source),
                str(target),
                label=data.get("type", ""),
                arrows="to"
            )

        # Save and embed in Streamlit
        output_file = "subgraph.html"
        net.write_html(output_file)

        with open(output_file, "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=650, scrolling=True)

        # Clean up
        os.remove(output_file)
