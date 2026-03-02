CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -msse2 -mavx
LDFLAGS = -lpthread -lm

TARGET = matmul-arena
SRC = src/matmul.c

.PHONY: all clean bench

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# quick benchmark at different sizes
bench: $(TARGET)
	@echo "=== Small (256x256) ==="
	./$(TARGET) 256
	@echo "=== Medium (512x512) ==="
	./$(TARGET) 512
	@echo "=== Large (1024x1024) ==="
	./$(TARGET) 1024

clean:
	rm -f $(TARGET)
