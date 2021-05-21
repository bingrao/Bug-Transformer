from py4j.java_gateway import JavaGateway, GatewayParameters

if __name__ == "__main__":
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=25333))

    input_data = "void method(String input) { Int a = b + 1; }"
    reg = gateway.entry_point.getAbstractCodeFromStringMethod(input_data, "idioms/idioms.csv")


    print(reg)
