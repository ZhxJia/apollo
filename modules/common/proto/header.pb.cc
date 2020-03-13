// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: header.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "header.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace apollo {
namespace common {

namespace {

const ::google::protobuf::Descriptor* Header_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  Header_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_header_2eproto() {
  protobuf_AddDesc_header_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "header.proto");
  GOOGLE_CHECK(file != NULL);
  Header_descriptor_ = file->message_type(0);
  static const int Header_offsets_[9] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Header, timestamp_sec_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Header, module_name_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Header, sequence_num_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Header, lidar_timestamp_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Header, camera_timestamp_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Header, radar_timestamp_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Header, version_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Header, status_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Header, frame_id_),
  };
  Header_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      Header_descriptor_,
      Header::default_instance_,
      Header_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Header, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Header, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(Header));
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_header_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    Header_descriptor_, &Header::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_header_2eproto() {
  delete Header::default_instance_;
  delete Header_reflection_;
}

void protobuf_AddDesc_header_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::apollo::common::protobuf_AddDesc_modules_2fcommon_2fproto_2ferror_5fcode_2eproto();
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\014header.proto\022\rapollo.common\032%modules/c"
    "ommon/proto/error_code.proto\"\345\001\n\006Header\022"
    "\025\n\rtimestamp_sec\030\001 \001(\001\022\023\n\013module_name\030\002 "
    "\001(\t\022\024\n\014sequence_num\030\003 \001(\r\022\027\n\017lidar_times"
    "tamp\030\004 \001(\004\022\030\n\020camera_timestamp\030\005 \001(\004\022\027\n\017"
    "radar_timestamp\030\006 \001(\004\022\022\n\007version\030\007 \001(\r:\001"
    "1\022\'\n\006status\030\010 \001(\0132\027.apollo.common.Status"
    "Pb\022\020\n\010frame_id\030\t \001(\t", 300);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "header.proto", &protobuf_RegisterTypes);
  Header::default_instance_ = new Header();
  Header::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_header_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_header_2eproto {
  StaticDescriptorInitializer_header_2eproto() {
    protobuf_AddDesc_header_2eproto();
  }
} static_descriptor_initializer_header_2eproto_;

// ===================================================================

#ifndef _MSC_VER
const int Header::kTimestampSecFieldNumber;
const int Header::kModuleNameFieldNumber;
const int Header::kSequenceNumFieldNumber;
const int Header::kLidarTimestampFieldNumber;
const int Header::kCameraTimestampFieldNumber;
const int Header::kRadarTimestampFieldNumber;
const int Header::kVersionFieldNumber;
const int Header::kStatusFieldNumber;
const int Header::kFrameIdFieldNumber;
#endif  // !_MSC_VER

Header::Header()
  : ::google::protobuf::Message() {
  SharedCtor();
  // @@protoc_insertion_point(constructor:apollo.common.Header)
}

void Header::InitAsDefaultInstance() {
  status_ = const_cast< ::apollo::common::StatusPb*>(&::apollo::common::StatusPb::default_instance());
}

Header::Header(const Header& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
  // @@protoc_insertion_point(copy_constructor:apollo.common.Header)
}

void Header::SharedCtor() {
  ::google::protobuf::internal::GetEmptyString();
  _cached_size_ = 0;
  timestamp_sec_ = 0;
  module_name_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  sequence_num_ = 0u;
  lidar_timestamp_ = GOOGLE_ULONGLONG(0);
  camera_timestamp_ = GOOGLE_ULONGLONG(0);
  radar_timestamp_ = GOOGLE_ULONGLONG(0);
  version_ = 1u;
  status_ = NULL;
  frame_id_ = const_cast< ::std::string*>(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

Header::~Header() {
  // @@protoc_insertion_point(destructor:apollo.common.Header)
  SharedDtor();
}

void Header::SharedDtor() {
  if (module_name_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    delete module_name_;
  }
  if (frame_id_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
    delete frame_id_;
  }
  if (this != default_instance_) {
    delete status_;
  }
}

void Header::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* Header::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return Header_descriptor_;
}

const Header& Header::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_header_2eproto();
  return *default_instance_;
}

Header* Header::default_instance_ = NULL;

Header* Header::New() const {
  return new Header;
}

void Header::Clear() {
#define OFFSET_OF_FIELD_(f) (reinterpret_cast<char*>(      \
  &reinterpret_cast<Header*>(16)->f) - \
   reinterpret_cast<char*>(16))

#define ZR_(first, last) do {                              \
    size_t f = OFFSET_OF_FIELD_(first);                    \
    size_t n = OFFSET_OF_FIELD_(last) - f + sizeof(last);  \
    ::memset(&first, 0, n);                                \
  } while (0)

  if (_has_bits_[0 / 32] & 255) {
    ZR_(lidar_timestamp_, sequence_num_);
    timestamp_sec_ = 0;
    if (has_module_name()) {
      if (module_name_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
        module_name_->clear();
      }
    }
    radar_timestamp_ = GOOGLE_ULONGLONG(0);
    version_ = 1u;
    if (has_status()) {
      if (status_ != NULL) status_->::apollo::common::StatusPb::Clear();
    }
  }
  if (has_frame_id()) {
    if (frame_id_ != &::google::protobuf::internal::GetEmptyStringAlreadyInited()) {
      frame_id_->clear();
    }
  }

#undef OFFSET_OF_FIELD_
#undef ZR_

  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool Header::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:apollo.common.Header)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoff(127);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // optional double timestamp_sec = 1;
      case 1: {
        if (tag == 9) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 input, &timestamp_sec_)));
          set_has_timestamp_sec();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(18)) goto parse_module_name;
        break;
      }

      // optional string module_name = 2;
      case 2: {
        if (tag == 18) {
         parse_module_name:
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_module_name()));
          ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
            this->module_name().data(), this->module_name().length(),
            ::google::protobuf::internal::WireFormat::PARSE,
            "module_name");
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(24)) goto parse_sequence_num;
        break;
      }

      // optional uint32 sequence_num = 3;
      case 3: {
        if (tag == 24) {
         parse_sequence_num:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &sequence_num_)));
          set_has_sequence_num();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(32)) goto parse_lidar_timestamp;
        break;
      }

      // optional uint64 lidar_timestamp = 4;
      case 4: {
        if (tag == 32) {
         parse_lidar_timestamp:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint64, ::google::protobuf::internal::WireFormatLite::TYPE_UINT64>(
                 input, &lidar_timestamp_)));
          set_has_lidar_timestamp();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(40)) goto parse_camera_timestamp;
        break;
      }

      // optional uint64 camera_timestamp = 5;
      case 5: {
        if (tag == 40) {
         parse_camera_timestamp:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint64, ::google::protobuf::internal::WireFormatLite::TYPE_UINT64>(
                 input, &camera_timestamp_)));
          set_has_camera_timestamp();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(48)) goto parse_radar_timestamp;
        break;
      }

      // optional uint64 radar_timestamp = 6;
      case 6: {
        if (tag == 48) {
         parse_radar_timestamp:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint64, ::google::protobuf::internal::WireFormatLite::TYPE_UINT64>(
                 input, &radar_timestamp_)));
          set_has_radar_timestamp();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(56)) goto parse_version;
        break;
      }

      // optional uint32 version = 7 [default = 1];
      case 7: {
        if (tag == 56) {
         parse_version:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::uint32, ::google::protobuf::internal::WireFormatLite::TYPE_UINT32>(
                 input, &version_)));
          set_has_version();
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(66)) goto parse_status;
        break;
      }

      // optional .apollo.common.StatusPb status = 8;
      case 8: {
        if (tag == 66) {
         parse_status:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
               input, mutable_status()));
        } else {
          goto handle_unusual;
        }
        if (input->ExpectTag(74)) goto parse_frame_id;
        break;
      }

      // optional string frame_id = 9;
      case 9: {
        if (tag == 74) {
         parse_frame_id:
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_frame_id()));
          ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
            this->frame_id().data(), this->frame_id().length(),
            ::google::protobuf::internal::WireFormat::PARSE,
            "frame_id");
        } else {
          goto handle_unusual;
        }
        if (input->ExpectAtEnd()) goto success;
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:apollo.common.Header)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:apollo.common.Header)
  return false;
#undef DO_
}

void Header::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:apollo.common.Header)
  // optional double timestamp_sec = 1;
  if (has_timestamp_sec()) {
    ::google::protobuf::internal::WireFormatLite::WriteDouble(1, this->timestamp_sec(), output);
  }

  // optional string module_name = 2;
  if (has_module_name()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
      this->module_name().data(), this->module_name().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE,
      "module_name");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      2, this->module_name(), output);
  }

  // optional uint32 sequence_num = 3;
  if (has_sequence_num()) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(3, this->sequence_num(), output);
  }

  // optional uint64 lidar_timestamp = 4;
  if (has_lidar_timestamp()) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt64(4, this->lidar_timestamp(), output);
  }

  // optional uint64 camera_timestamp = 5;
  if (has_camera_timestamp()) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt64(5, this->camera_timestamp(), output);
  }

  // optional uint64 radar_timestamp = 6;
  if (has_radar_timestamp()) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt64(6, this->radar_timestamp(), output);
  }

  // optional uint32 version = 7 [default = 1];
  if (has_version()) {
    ::google::protobuf::internal::WireFormatLite::WriteUInt32(7, this->version(), output);
  }

  // optional .apollo.common.StatusPb status = 8;
  if (has_status()) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      8, this->status(), output);
  }

  // optional string frame_id = 9;
  if (has_frame_id()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
      this->frame_id().data(), this->frame_id().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE,
      "frame_id");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      9, this->frame_id(), output);
  }

  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:apollo.common.Header)
}

::google::protobuf::uint8* Header::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:apollo.common.Header)
  // optional double timestamp_sec = 1;
  if (has_timestamp_sec()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteDoubleToArray(1, this->timestamp_sec(), target);
  }

  // optional string module_name = 2;
  if (has_module_name()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
      this->module_name().data(), this->module_name().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE,
      "module_name");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        2, this->module_name(), target);
  }

  // optional uint32 sequence_num = 3;
  if (has_sequence_num()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(3, this->sequence_num(), target);
  }

  // optional uint64 lidar_timestamp = 4;
  if (has_lidar_timestamp()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt64ToArray(4, this->lidar_timestamp(), target);
  }

  // optional uint64 camera_timestamp = 5;
  if (has_camera_timestamp()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt64ToArray(5, this->camera_timestamp(), target);
  }

  // optional uint64 radar_timestamp = 6;
  if (has_radar_timestamp()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt64ToArray(6, this->radar_timestamp(), target);
  }

  // optional uint32 version = 7 [default = 1];
  if (has_version()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteUInt32ToArray(7, this->version(), target);
  }

  // optional .apollo.common.StatusPb status = 8;
  if (has_status()) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        8, this->status(), target);
  }

  // optional string frame_id = 9;
  if (has_frame_id()) {
    ::google::protobuf::internal::WireFormat::VerifyUTF8StringNamedField(
      this->frame_id().data(), this->frame_id().length(),
      ::google::protobuf::internal::WireFormat::SERIALIZE,
      "frame_id");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        9, this->frame_id(), target);
  }

  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:apollo.common.Header)
  return target;
}

int Header::ByteSize() const {
  int total_size = 0;

  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    // optional double timestamp_sec = 1;
    if (has_timestamp_sec()) {
      total_size += 1 + 8;
    }

    // optional string module_name = 2;
    if (has_module_name()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::StringSize(
          this->module_name());
    }

    // optional uint32 sequence_num = 3;
    if (has_sequence_num()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt32Size(
          this->sequence_num());
    }

    // optional uint64 lidar_timestamp = 4;
    if (has_lidar_timestamp()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt64Size(
          this->lidar_timestamp());
    }

    // optional uint64 camera_timestamp = 5;
    if (has_camera_timestamp()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt64Size(
          this->camera_timestamp());
    }

    // optional uint64 radar_timestamp = 6;
    if (has_radar_timestamp()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt64Size(
          this->radar_timestamp());
    }

    // optional uint32 version = 7 [default = 1];
    if (has_version()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::UInt32Size(
          this->version());
    }

    // optional .apollo.common.StatusPb status = 8;
    if (has_status()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
          this->status());
    }

  }
  if (_has_bits_[8 / 32] & (0xffu << (8 % 32))) {
    // optional string frame_id = 9;
    if (has_frame_id()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::StringSize(
          this->frame_id());
    }

  }
  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void Header::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const Header* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const Header*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void Header::MergeFrom(const Header& from) {
  GOOGLE_CHECK_NE(&from, this);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from.has_timestamp_sec()) {
      set_timestamp_sec(from.timestamp_sec());
    }
    if (from.has_module_name()) {
      set_module_name(from.module_name());
    }
    if (from.has_sequence_num()) {
      set_sequence_num(from.sequence_num());
    }
    if (from.has_lidar_timestamp()) {
      set_lidar_timestamp(from.lidar_timestamp());
    }
    if (from.has_camera_timestamp()) {
      set_camera_timestamp(from.camera_timestamp());
    }
    if (from.has_radar_timestamp()) {
      set_radar_timestamp(from.radar_timestamp());
    }
    if (from.has_version()) {
      set_version(from.version());
    }
    if (from.has_status()) {
      mutable_status()->::apollo::common::StatusPb::MergeFrom(from.status());
    }
  }
  if (from._has_bits_[8 / 32] & (0xffu << (8 % 32))) {
    if (from.has_frame_id()) {
      set_frame_id(from.frame_id());
    }
  }
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void Header::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Header::CopyFrom(const Header& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Header::IsInitialized() const {

  return true;
}

void Header::Swap(Header* other) {
  if (other != this) {
    std::swap(timestamp_sec_, other->timestamp_sec_);
    std::swap(module_name_, other->module_name_);
    std::swap(sequence_num_, other->sequence_num_);
    std::swap(lidar_timestamp_, other->lidar_timestamp_);
    std::swap(camera_timestamp_, other->camera_timestamp_);
    std::swap(radar_timestamp_, other->radar_timestamp_);
    std::swap(version_, other->version_);
    std::swap(status_, other->status_);
    std::swap(frame_id_, other->frame_id_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata Header::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = Header_descriptor_;
  metadata.reflection = Header_reflection_;
  return metadata;
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace common
}  // namespace apollo

// @@protoc_insertion_point(global_scope)
