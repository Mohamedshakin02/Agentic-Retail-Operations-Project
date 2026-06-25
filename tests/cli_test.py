

from agent_graph import find_top_risk_products
import agent_graph as ag

print("\n=== Interactive mode — type 'quit' to stop ===")
while True:
    user_input = input("\nTry 'S003 P001' or 'top 5': ").strip()
    if user_input.lower() == "quit":
        break
    parts = user_input.split()
    if parts and parts[0].lower() == "top":
        n = int(parts[1]) if len(parts) > 1 else 5
        for item in find_top_risk_products(n=n):
            print(f"\n{item['product_id']} @ {item['store_id']}: {item['summary']}")
    elif len(parts) == 2:
        result = retail_agent.invoke({"store_id": parts[0], "product_id": parts[1]})
        print("\n", result["summary"])
    else:
        print("Format: 'store_id product_id' or 'top N'")