"""Minimal macOS generic-password bridge; key bytes never enter argv or files."""

from __future__ import annotations
import ctypes as C
import hashlib
import sys


class KeychainUnavailable(RuntimeError):
    pass


class KeychainKeys:
    def __init__(self, namespace):
        self.service = (
            "org.local.facial-analysis.history."
            + hashlib.sha256(str(namespace).encode()).hexdigest()[:24]
        )
        self.account = "archive-aes256-v1"

    def _libraries(self):
        if sys.platform != "darwin":
            raise KeychainUnavailable("Encrypted history requires macOS Keychain.")
        cf = C.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        sec = C.CDLL("/System/Library/Frameworks/Security.framework/Security")
        cf.CFStringCreateWithCString.argtypes = [C.c_void_p, C.c_char_p, C.c_uint32]
        cf.CFStringCreateWithCString.restype = C.c_void_p
        cf.CFDataCreate.argtypes = [C.c_void_p, C.c_void_p, C.c_long]
        cf.CFDataCreate.restype = C.c_void_p
        cf.CFDictionaryCreate.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_void_p),
            C.c_long,
            C.c_void_p,
            C.c_void_p,
        ]
        cf.CFDictionaryCreate.restype = C.c_void_p
        cf.CFRelease.argtypes = [C.c_void_p]
        cf.CFDataGetLength.argtypes = [C.c_void_p]
        cf.CFDataGetLength.restype = C.c_long
        cf.CFDataGetBytePtr.argtypes = [C.c_void_p]
        cf.CFDataGetBytePtr.restype = C.c_void_p
        # SecItemAdd on the legacy macOS Keychain may ignore kSecUseAuthenticationUIFail.
        # Disable interaction for this process before any lookup/create; never ask
        # macOS to create/reset a Keychain when the configured default is absent.
        sec.SecKeychainSetUserInteractionAllowed.argtypes = [C.c_ubyte]
        sec.SecKeychainSetUserInteractionAllowed.restype = C.c_int32
        if sec.SecKeychainSetUserInteractionAllowed(False) != 0:
            raise KeychainUnavailable(
                "Keychain interaction could not be disabled; history remains locked."
            )
        sec.SecKeychainCopyDefault.argtypes = [C.POINTER(C.c_void_p)]
        sec.SecKeychainCopyDefault.restype = C.c_int32
        sec.SecKeychainGetStatus.argtypes = [C.c_void_p, C.POINTER(C.c_uint32)]
        sec.SecKeychainGetStatus.restype = C.c_int32
        default = C.c_void_p()
        status = sec.SecKeychainCopyDefault(C.byref(default))
        if status != 0 or not default.value:
            raise KeychainUnavailable(
                "No default macOS Keychain is available. History stays locked; no Keychain was created or reset."
            )
        try:
            flags = C.c_uint32()
            if (
                sec.SecKeychainGetStatus(default, C.byref(flags)) != 0
                or not flags.value & 1
            ):
                raise KeychainUnavailable(
                    "The default Keychain is unavailable or locked. Unlock it in macOS before saving history."
                )
        finally:
            cf.CFRelease(default)
        sec.SecItemCopyMatching.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)]
        sec.SecItemCopyMatching.restype = C.c_int32
        sec.SecItemAdd.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)]
        sec.SecItemAdd.restype = C.c_int32
        sec.SecItemDelete.argtypes = [C.c_void_p]
        sec.SecItemDelete.restype = C.c_int32
        return cf, sec

    def _operation(self, action, key=None):
        cf, sec = self._libraries()
        owned = []

        def symbol(name):
            return C.c_void_p.in_dll(sec, name).value

        def string(value):
            obj = cf.CFStringCreateWithCString(None, value.encode(), 0x08000100)
            owned.append(obj)
            return obj

        pairs = [
            (symbol("kSecClass"), symbol("kSecClassGenericPassword")),
            (symbol("kSecAttrService"), string(self.service)),
            (symbol("kSecAttrAccount"), string(self.account)),
            (symbol("kSecUseAuthenticationUI"), symbol("kSecUseAuthenticationUIFail")),
        ]
        if action == "get":
            pairs.append(
                (
                    symbol("kSecReturnData"),
                    C.c_void_p.in_dll(cf, "kCFBooleanTrue").value,
                )
            )
        if action == "create":
            data = cf.CFDataCreate(None, key, len(key))
            owned.append(data)
            pairs.extend(
                [
                    (symbol("kSecValueData"), data),
                    (
                        symbol("kSecAttrAccessible"),
                        symbol("kSecAttrAccessibleWhenUnlockedThisDeviceOnly"),
                    ),
                ]
            )
        keys = (C.c_void_p * len(pairs))(*(p[0] for p in pairs))
        values = (C.c_void_p * len(pairs))(*(p[1] for p in pairs))
        query = cf.CFDictionaryCreate(None, keys, values, len(pairs), None, None)
        owned.append(query)
        result = C.c_void_p()
        try:
            if action == "get":
                status = sec.SecItemCopyMatching(query, C.byref(result))
            elif action == "create":
                status = sec.SecItemAdd(query, None)
            else:
                status = sec.SecItemDelete(query)
            if action == "get" and status == -25300:
                return None
            if status != 0:
                raise KeychainUnavailable(
                    "Keychain is locked, denied or unavailable; history remains encrypted."
                )
            if action == "get":
                if cf.CFDataGetLength(result) != 32:
                    raise KeychainUnavailable(
                        "The archive key is malformed; history remains locked."
                    )
                return C.string_at(cf.CFDataGetBytePtr(result), 32)
            return None
        finally:
            if result.value:
                cf.CFRelease(result)
            for obj in reversed(owned):
                if obj:
                    cf.CFRelease(obj)

    def get(self):
        return self._operation("get")

    def create(self, key):
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError("A 256-bit key is required.")
        self._operation("create", key)

    def delete(self):
        self._operation("delete")
