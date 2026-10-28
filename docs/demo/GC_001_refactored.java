import java.util.*;

public class OrderProcessor {
    private final OrderValidator validator;
    private final PaymentService paymentService;
    private final InventoryService inventoryService;
    private final NotificationService notificationService;
    private final ShippingService shippingService;
    private final OrderRepository orderRepository;

    public OrderProcessor() {
        this.validator = new OrderValidator();
        this.paymentService = new PaymentService();
        this.inventoryService = new InventoryService();
        this.notificationService = new NotificationService();
        this.shippingService = new ShippingService();
        this.orderRepository = new OrderRepository();
    }

    // Validation
    public boolean validateOrder(Order order) {
        return validator.validate(order);
    }

    // Payment
    public boolean processPayment(Order order, PaymentDetails pd) {
        return paymentService.processPayment(order, pd);
    }

    // Inventory
    public boolean checkInventory(Order order) {
        return inventoryService.checkInventory(order);
    }

    public void reserveInventory(Order order) {
        inventoryService.reserveInventory(order);
    }

    // Notification
    public void sendEmail(String to, String subject, String body) {
        notificationService.sendEmail(to, subject, body);
    }

    // Shipping
    public String createShippingLabel(Order order) {
        return shippingService.createShippingLabel(order);
    }

    // Orchestration
    public boolean processOrder(Order order, PaymentDetails pd) {
        if (!validateOrder(order)) return false;
        if (!checkInventory(order)) return false;
        if (!processPayment(order, pd)) return false;
        reserveInventory(order);
        sendEmail(order.getCustomerEmail(), "Confirmation", "Done");
        createShippingLabel(order);
        orderRepository.add(order);
        return true;
    }

    public void addStock(String sku, int quantity) {
        inventoryService.addStock(sku, quantity);
    }

    public double getTotalRevenue() { return paymentService.getTotalRevenue(); }
    public int getProcessedOrderCount() { return orderRepository.count(); }

    public static class Item {
        private final String sku;
        private final int quantity;
        public Item(String sku, int quantity) { this.sku = sku; this.quantity = quantity; }
        public String getSku() { return sku; }
        public int getQuantity() { return quantity; }
    }

    public static class Order {
        private final String id, customerId, customerEmail;
        private final List<Item> items;
        private final double total;
        public Order(String id, String customerId, String customerEmail,
                     List<Item> items, double total) {
            this.id = id; this.customerId = customerId;
            this.customerEmail = customerEmail; this.items = items; this.total = total;
        }
        public String getId() { return id; }
        public String getCustomerId() { return customerId; }
        public String getCustomerEmail() { return customerEmail; }
        public List<Item> getItems() { return items; }
        public double getTotal() { return total; }
    }

    public static class PaymentDetails {
        private final double amount;
        public PaymentDetails(double amount) { this.amount = amount; }
        public double getAmount() { return amount; }
    }
}

class OrderValidator {
    boolean validate(OrderProcessor.Order order) {
        if (order == null) return false;
        if (order.getItems().isEmpty()) return false;
        if (order.getCustomerId() == null) return false;
        return true;
    }
}

class PaymentService {
    private double totalRevenue = 0.0;

    boolean processPayment(OrderProcessor.Order order, OrderProcessor.PaymentDetails pd) {
        if (pd.getAmount() < order.getTotal()) return false;
        totalRevenue += order.getTotal();
        return true;
    }

    double getTotalRevenue() {
        return totalRevenue;
    }
}

class InventoryService {
    private final Map<String, Integer> inventory = new HashMap<>();

    boolean checkInventory(OrderProcessor.Order order) {
        for (OrderProcessor.Item item : order.getItems()) {
            int stock = inventory.getOrDefault(item.getSku(), 0);
            if (stock < item.getQuantity()) return false;
        }
        return true;
    }

    void reserveInventory(OrderProcessor.Order order) {
        for (OrderProcessor.Item item : order.getItems()) {
            inventory.merge(item.getSku(), -item.getQuantity(), Integer::sum);
        }
    }

    void addStock(String sku, int quantity) {
        inventory.merge(sku, quantity, Integer::sum);
    }
}

class NotificationService {
    void sendEmail(String to, String subject, String body) {
        System.out.println("Email to: " + to);
    }
}

class ShippingService {
    String createShippingLabel(OrderProcessor.Order order) {
        return "SHIP-" + order.getId() + "-" + System.currentTimeMillis();
    }
}

class OrderRepository {
    private final List<OrderProcessor.Order> processedOrders = new ArrayList<>();

    void add(OrderProcessor.Order order) {
        processedOrders.add(order);
    }

    int count() {
        return processedOrders.size();
    }
}