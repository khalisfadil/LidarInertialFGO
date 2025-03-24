#pragma once

#include <cstdint>
#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>
#include <boost/asio.hpp>

namespace slam {
    class UDPSocket {
    public:
        UDPSocket(boost::asio::io_context& context,
                  const std::string& host,
                  uint16_t port,
                  std::function<void(const std::vector<uint8_t>&)> callback,
                  uint32_t bufferSize = 65535)
            : socket_(context), callback_(std::move(callback)) {
            boost::asio::ip::udp::resolver resolver(context);
            boost::asio::ip::udp::resolver::results_type endpoints =
                resolver.resolve(boost::asio::ip::udp::v4(), host, std::to_string(port));
            socket_.open(boost::asio::ip::udp::v4());
            socket_.bind(*endpoints.begin());
            buffer_.resize(bufferSize);  // Pre-allocate buffer memory
            startReceive();              // Begin listening for packets
        }

        ~UDPSocket() {
            stop();
        }

        void startReceive() {
            if (!isRunning_) {
                return;
            }

            socket_.async_receive_from(
                boost::asio::buffer(buffer_),
                senderEndpoint_,
                [this](boost::system::error_code ec, std::size_t bytesReceived) {
                    if (!ec && bytesReceived > 0) {
                        std::vector<uint8_t> packetData(buffer_.begin(), buffer_.begin() + bytesReceived);

#ifdef DEBUG
                        std::cout << "Received packet (size: " << bytesReceived << "): ";
                        for (size_t i = 0; i < bytesReceived; ++i) {
                            std::cout << std::hex << std::setw(2) << std::setfill('0')
                                      << static_cast<int>(packetData[i]) << " ";
                        }
                        std::cout << std::dec << '\n';
#endif

                        if (callback_) {
                            callback_(std::move(packetData));
                        }
                    } else if (ec == boost::asio::error::operation_aborted) {
                        return;
                    } else if (ec) {
                        std::cerr << "Receive error: " << ec.message() << '\n';
                    }

                    if (isRunning_) {
                        startReceive();
                    }
                });
        }

        void stop() {
            if (!isRunning_) {
                return;
            }

            isRunning_ = false;
            boost::system::error_code ec;
            socket_.cancel(ec);
            if (ec) {
                std::cerr << "Error cancelling operations: " << ec.message() << '\n';
            }
            socket_.close(ec);
            if (ec) {
                std::cerr << "Error closing socket: " << ec.message() << '\n';
            }
        }

        void setReceiveBufferSize(uint32_t size) {
            boost::system::error_code ec;
            boost::asio::socket_base::receive_buffer_size option(size);
            socket_.set_option(option, ec);
            if (ec) {
                std::cerr << "Error setting receive buffer size: " << ec.message() << '\n';
            } else {
                std::cout << "Receive buffer size set to " << size << " bytes\n";
            }
        }

    private:
        boost::asio::ip::udp::socket socket_;
        boost::asio::ip::udp::endpoint senderEndpoint_;
        std::vector<uint8_t> buffer_;
        std::function<void(const std::vector<uint8_t>&)> callback_;
        bool isRunning_ = true;
    };

}  // namespace slam