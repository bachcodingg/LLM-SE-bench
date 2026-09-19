package com.example.benchmark;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

// Simple shopping cart
public class Cart {

    private List<Map<String, Object>> items;
    private double tax;
    private String custName;
    private boolean memberDiscount;
    private Map<String, Integer> stock;

    public Cart(String name, boolean isMember) {
        this.items = new ArrayList<>();
        this.tax = 0.08;
        this.custName = name;
        this.memberDiscount = isMember;
        this.stock = new HashMap<>();
    }

    public void setStock(Map<String, Integer> s) {
        this.stock = s;
    }

    public boolean addItem(String name, double price, int qty) {
        if (price < 0 || qty <= 0) return false;
        if (stock.containsKey(name)) {
            if (stock.get(name) < qty) return false;
            stock.put(name, stock.get(name) - qty);
        }
        Map<String, Object> item = new HashMap<>();
        item.put("name", name);
        item.put("price", price);
        item.put("qty", qty);
        items.add(item);
        return true;
    }

    public boolean removeItem(String name) {
        for (int i = 0; i < items.size(); i++) {
            if (items.get(i).get("name").equals(name)) {
                Map<String, Object> removed = items.remove(i);
                if (stock.containsKey(name)) {
                    stock.put(name, stock.get(name) + (int) removed.get("qty"));
                }
                return true;
            }
        }
        return false;
    }

    public double calcSubtotal() {
        double sub = 0;
        for (Map<String, Object> item : items) {
            sub += (double) item.get("price") * (int) item.get("qty");
        }
        return sub;
    }

    public double calcTotal() {
        double sub = calcSubtotal();
        double t = sub * tax;
        double total = sub + t;
        if (memberDiscount && sub > 50) {
            total = total * 0.9;
        } else if (memberDiscount) {
            total = total * 0.95;
        }
        return Math.round(total * 100.0) / 100.0;
    }

    // get receipt
    public String getReceipt() {
        StringBuilder sb = new StringBuilder();
        sb.append("Customer: " + custName + "\n");
        sb.append("---\n");
        for (Map<String, Object> item : items) {
            sb.append(item.get("name") + " x" + item.get("qty") + " @ $" + item.get("price") + "\n");
        }
        sb.append("---\n");
        sb.append("Subtotal: $" + calcSubtotal() + "\n");
        sb.append("Tax: $" + Math.round(calcSubtotal() * tax * 100.0) / 100.0 + "\n");
        if (memberDiscount) sb.append("Member discount applied\n");
        sb.append("Total: $" + calcTotal() + "\n");
        return sb.toString();
    }

    public int getItemCount() {
        int c = 0;
        for (Map<String, Object> item : items) {
            c += (int) item.get("qty");
        }
        return c;
    }

    public List<Map<String, Object>> getItems() { return items; }
    public String getCustName() { return custName; }
}
